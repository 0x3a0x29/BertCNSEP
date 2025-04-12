import torch.utils
import torch.utils.data
from transformers import BertTokenizer
from transformers import BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn.functional as F
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# ========================== 超参数 ==========================
BATCH_SIZE = 10 
NUM_WORKERS = 4
NUM_EPOCHS = 10
LR = 1e-5
SAVE_PATH='finetuned_models/trained_model.pth'
BERT_MAX_LENGTH = 400
SAVE_MODEL = True
SAVE_PIC = True

# ========================== 数据集类 ==========================
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.dataset = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                split_line = line.strip().split()  
                first = split_line[1:8]  
                try:
                    totals = int(first[0][6:])  
                except ValueError:
                    continue  
                if totals == 0:
                    emotions = [0] * 6  
                else:
                    emotions = [int(first[i+1][3:]) / totals for i in range(6)] #最佳的概率分布  
                remaining = "".join(split_line[8:])
                self.dataset.append({'text': remaining, 'label': emotions})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        return text, label

# 加载分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 读取数据集

dataset = Dataset('data/2016.1-2016.11')
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# ========================== 数据加载器 ==========================
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    encoded_data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='max_length',
        max_length=BERT_MAX_LENGTH,
        return_tensors='pt',
        return_length=True
    )
    input_ids = encoded_data['input_ids']  
    attention_mask = encoded_data['attention_mask']  
    token_type_ids = encoded_data['token_type_ids']  
    labels = torch.FloatTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn, shuffle=False)

# ========================== BERT 模型定义 ==========================
class BertClassifier(nn.Module):
    def __init__(self, bert_model, dropout=0.4, hidden_dim=128, num_classes=6):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.mlp = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.mlp(pooled_output)
        probabilities = F.softmax(logits, dim=1)  # 对logits使用softmax函数转成概率分布
        return probabilities

# 初始化模型
pretrained = BertModel.from_pretrained('bert-base-chinese')
model = BertClassifier(pretrained).to(device)
# 损失函数 & 优化器
criterion = torch.nn.KLDivLoss(reduction = 'batchmean')
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

# ========================== 评估函数 ==========================
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pearson_sum = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in loader:
            input_ids, attention_mask, token_type_ids, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device),
                labels.to(device),
            )

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs.log(), labels)
            #loss = js_divergence_loss(outputs,labels)
            total_loss += loss.item()
            pred_labels = torch.argmax(outputs, dim=1)  # 计算预测的Acc@1
            gold_labels = torch.argmax(labels, dim=1)  # 计算实际的Acc@1
            total_correct += (pred_labels == gold_labels).sum().item()
            total_samples += labels.size(0)
            
            for i in range(labels.size(0)):
                label_vec=labels[i].cpu().detach().numpy()
                output_vec=outputs[i].cpu().detach().numpy()
                pearson_corr, _ = pearsonr(label_vec,output_vec)
                pearson_sum+=pearson_corr
    avg_loss = total_loss / len(loader)
    accuracy1 = total_correct / total_samples
    avg_pearson = pearson_sum/total_samples
    return avg_loss, accuracy1, avg_pearson

# ========================== 训练过程 ==========================

train_losses, test_losses, train_accuracies, test_accuracies, train_aps, test_aps = [], [], [], [], [], []
best_test_loss = np.inf
start = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    correct_train = 0
    total_train = 0
    pearson_sum = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch") as tepoch:
        for input_ids, attention_mask, token_type_ids, labels in tepoch:
            input_ids, attention_mask, token_type_ids, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                token_type_ids.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs.log(), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred_labels = torch.argmax(outputs, dim=1)
            gold_labels = torch.argmax(labels, dim=1)
            correct_train += (pred_labels == gold_labels).sum().item()
            total_train += labels.size(0)
            for i in range(labels.size(0)):
                label_vec=labels[i].cpu().detach().numpy()
                output_vec=outputs[i].cpu().detach().numpy()
                pearson_corr, _ = pearsonr(label_vec,output_vec)
                pearson_sum+=pearson_corr
            tepoch.set_postfix(loss=loss.item())

    train_accuracy = correct_train / total_train
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_pearson = pearson_sum / total_train
    avg_test_loss, test_accuracy, avg_test_pearson = evaluate(model, test_loader)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    train_aps.append(avg_train_pearson)
    test_aps.append(avg_test_pearson)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Train Acc@1: {train_accuracy:.4f} Train AP: {avg_train_pearson:.4f} | | Test Loss: {avg_test_loss:.4f} | Test Acc@1: {test_accuracy:.4f} | Test AP: {avg_test_pearson:.4f}")
    if (avg_test_loss < best_test_loss) and (SAVE_MODEL==True):
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"获得在测试集上表现更加优秀的模型，已保存至 {SAVE_PATH}")
end = time.time()
print(f"\n训练完成，总耗时：{end - start:.2f} 秒")
# ========================== 训练结果可视化 ==========================
if SAVE_PIC==True:
    plt.figure(figsize=(12, 6))

    # 训练损失 & 测试损失
    plt.subplot(2, 2, 1)
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), test_losses, label='Test Loss', color='red', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.grid(True)

    # 训练准确率 & 测试准确率
    plt.subplot(2, 2, 2)
    plt.plot(range(1, NUM_EPOCHS+1), train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), test_accuracies, label='Test Accuracy', color='red', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.grid(True)

    # 训练AP & 测试AP
    plt.subplot(2, 2, 3)
    plt.plot(range(1, NUM_EPOCHS+1), train_aps, label='Train AP', color='blue', marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), test_aps, label='Test AP', color='red', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('AP')
    plt.title('Train and Test AP (Pearson Correlation)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('evaluation.png')
    print("训练过程图已保存！")

torch.cuda.empty_cache()

# nvidia-smi
# ps -ef|grep [pid]
# kill -9 [pid]