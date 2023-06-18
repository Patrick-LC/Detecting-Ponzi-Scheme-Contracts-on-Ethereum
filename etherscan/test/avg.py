import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 读取数据
data = pd.read_csv('/Users/lc/Desktop/etherscan/transaction_dataset.csv')  # 替换为您的数据路径

# 选择特征
selected_features = ['Avg min between sent tnx', 'Avg min between received tnx',
                     'Time Diff between first and last (Mins)',
                     'Sent tnx', 'Received Tnx', 'Number of Created Contracts',
                     'Unique Received From Addresses', 'Unique Sent To Addresses',
                     'min value received', 'max value received', 'avg val received',
                     'min val sent', 'max val sent', 'avg val sent',
                     'min value sent to contract', 'max val sent to contract',
                     'avg value sent to contract', 'total transactions (including tnx to create contract)',
                     'total Ether sent', 'total ether received', 'total ether sent contracts',
                     'total ether balance', ' Total ERC20 tnxs',
                     ' ERC20 total Ether received', ' ERC20 total ether sent',
                     ' ERC20 total Ether sent contract', ' ERC20 uniq sent addr',
                     ' ERC20 uniq rec addr', ' ERC20 uniq sent addr.1',
                     ' ERC20 uniq rec contract addr', ' ERC20 avg time between sent tnx',
                     ' ERC20 avg time between rec tnx', ' ERC20 avg time between rec 2 tnx',
                     ' ERC20 avg time between contract tnx', ' ERC20 min val rec',
                     ' ERC20 max val rec', ' ERC20 avg val rec', ' ERC20 min val sent',
                     ' ERC20 max val sent', ' ERC20 avg val sent', ' ERC20 min val sent contract',
                     ' ERC20 max val sent contract', ' ERC20 avg val sent contract',
                     ' ERC20 uniq sent token name', ' ERC20 uniq rec token name']

data = data[selected_features + ['FLAG']]

# 数据预处理
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X = imputer.fit_transform(data.drop('FLAG', axis=1))
Y = data['FLAG']

scaler = StandardScaler()  # 数据标准化
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 神经网络分类器
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 评估模型
def evaluate_model(model, X, Y):
    pred = model.predict(X)
    pred = np.round(pred).flatten()
    accuracy = accuracy_score(Y, pred)
    recall = recall_score(Y, pred)
    f1 = f1_score(Y, pred)
    return accuracy, recall, f1

# 连续训练10次并记录评估指标
accuracies = []
recalls = []
f1_scores = []
for i in range(10):
    print("Training Model", i+1)
    model = create_model()

    # 定义EarlyStopping和ModelCheckpoint回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # 训练神经网络模型
    history = model.fit(X_train, Y_train, batch_size=64, epochs=140, verbose=0,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stopping, model_checkpoint])

    # 加载在验证集上性能最好的模型
    model.load_weights('best_model.h5')

    # 在测试集上评估模型
    accuracy, recall, f1 = evaluate_model(model, X_test, Y_test)
    accuracies.append(accuracy)
    recalls.append(recall)
    f1_scores.append(f1)

# 构建评估指标表格
results = pd.DataFrame({'Model': range(1, 11), 'Accuracy': accuracies, 'Recall': recalls, 'F1 Score': f1_scores})
results = results.sort_values(by=['Model'], ascending=True)

# 绘制表格图
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.set_style("white")
ax = sns.heatmap(results[['Accuracy', 'Recall', 'F1 Score']].T, annot=True, fmt=".4f", cmap="binary", cbar=False, linewidths=0.5, linecolor='black')
plt.xlabel('Model')
plt.ylabel('Metrics')
plt.title('Performance Metrics for 10 Models')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()
