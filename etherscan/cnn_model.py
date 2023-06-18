import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义自定义数据集类
class TransactionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['FLAG'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transaction = self.data.iloc[idx, 2:].values

        # 数据填充处理
        transaction = self.fill_missing_values(transaction)

        # 获取标签
        label = self.labels[idx]

        return transaction, label

    def fill_missing_values(self, transaction):
        transaction = np.where(transaction == ' ', '0', transaction)  # 将空字符串替换为0
        transaction = np.where(np.char.isdigit(transaction.astype(str)), transaction, '0')  # 非数值字符串替换为0
        transaction = transaction.astype('float32')
        mean_value = np.mean(transaction[np.nonzero(transaction)])  # 计算均值，排除0值
        transaction = np.where(transaction == 0, mean_value, transaction)  # 将0值替换为均值
        return transaction

def train_and_evaluate_cnn():
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
    num_epochs = 60
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
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = batch.unsqueeze(1).to(device)
                labels = labels.to(device)

                outputs = model(batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_precision = precision_score(val_targets, val_predictions)
        val_recall = recall_score(val_targets, val_predictions)
        val_f1_score = f1_score(val_targets, val_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Precision: {val_precision}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Recall: {val_recall}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation F1 Score: {val_f1_score}")

    # 在测试集上评估模型
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_precision = precision_score(test_targets, test_predictions)
    test_recall = recall_score(test_targets, test_predictions)
    test_f1_score = f1_score(test_targets, test_predictions)

    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1_score)

    return {
        "Accuracy": test_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1 Score": test_f1_score
    }

if __name__ == "__main__":
    train_and_evaluate_cnn()
