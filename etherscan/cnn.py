import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 定义自定义数据集类
class TransactionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, 2:].values
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['FLAG'])

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.labels = self.onehot_encoder.fit_transform(self.labels.reshape(-1, 1))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            features = self.features[idx]
            label = self.labels[idx]
            if features is None or label is None:
                raise ValueError("Invalid sample encountered with index: {}".format(idx))
            return features, label

    
    def fill_missing_values(self, transaction):
        transaction = np.where(transaction == ' ', np.nan, transaction)
        transaction = np.where(~np.char.isdigit(transaction.astype(str)), 'special_value', transaction)
        transaction = np.where(transaction == 'special_value', 1, transaction)
        transaction = transaction.astype(float)
        transaction[np.isnan(transaction)] = np.nanmean(transaction)
        return transaction



# 加载数据集
dataset = TransactionDataset('/Users/lc/Graduation Thesis/etherscan/transaction_dataset.csv')

# 拆分数据集为训练集、验证集和测试集
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义你的CNN模型的结构
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 24, 64)
        self.fc2 = nn.Linear(64, 2)

        # 将模型参数转换为Double类型
        self.conv1.double()
        self.fc1.double()
        self.fc2.double()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 初始化CNN模型
model = CNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练CNN模型
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for batch, labels in train_loader:
        batch = batch.unsqueeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(batch)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {correct / total}")

# 在测试集上评估模型
model.eval()
test_loss = 0.0
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for batch, labels in test_loader:
        batch = batch.unsqueeze(1).to(device)
        labels = labels.to(device)

        outputs = model(batch)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
confusion = confusion_matrix(true_labels, predicted_labels)

# 绘制混淆矩阵图
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 计算召回率、准确率和 F1 值
tn, fp, fn, tp = confusion.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# 输出召回率、准确率和 F1 值的表格
results = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Value": [accuracy, precision, recall, f1_score]})
print(results)
