import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense

# 读取数据
data = pd.read_csv('/Users/lc/Desktop/etherscan/transaction_dataset.csv') 
'''
# 选择特征
selected_features = ['Avg min between sent tnx', 'Avg min between received tnx', 'Time Diff between first and last (Mins)',
                     'Sent tnx', 'Received Tnx', 'Number of Created Contracts', 'Unique Received From Addresses',
                     'Unique Sent To Addresses', 'min value received', 'max value received', 'avg val received',
                     'min val sent', 'max val sent', 'avg val sent', 'total transactions (including tnx to create contract)',
                     'total Ether sent', 'total ether received', 'total ether sent contracts', 'total ether balance']
'''
selected_features = ['Avg min between sent tnx','Avg min between received tnx','Time Diff between first and last (Mins)',
                     'Sent tnx','Received Tnx','Number of Created Contracts','Unique Received From Addresses','Unique Sent To Addresses',
                     'min value received','max value received','avg val received','min val sent','max val sent','avg val sent',
                     'min value sent to contract','max val sent to contract','avg value sent to contract','total transactions (including tnx to create contract)',
                     'total Ether sent','total ether received','total ether sent contracts','total ether balance',' Total ERC20 tnxs',
                     ' ERC20 total Ether received',' ERC20 total ether sent',' ERC20 total Ether sent contract',' ERC20 uniq sent addr',
                     ' ERC20 uniq rec addr',' ERC20 uniq sent addr.1',' ERC20 uniq rec contract addr',' ERC20 avg time between sent tnx',
                     ' ERC20 avg time between rec tnx',' ERC20 avg time between rec 2 tnx',' ERC20 avg time between contract tnx',' ERC20 min val rec',
                     ' ERC20 max val rec',' ERC20 avg val rec',' ERC20 min val sent',' ERC20 max val sent',' ERC20 avg val sent',' ERC20 min val sent contract',
                     ' ERC20 max val sent contract',' ERC20 avg val sent contract',' ERC20 uniq sent token name',' ERC20 uniq rec token name']

data = data[selected_features + ['FLAG']]

# 数据预处理
# TODO: 执行适当的数据处理步骤，包括特征选择、缺失值处理、数据转换等

# 划分特征和标签
X = data.drop('FLAG', axis=1)
Y = data['FLAG']

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 随机森林分类器
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)
rfc_accuracy = accuracy_score(Y_test, rfc_pred)
print("Random Forest Classifier Accuracy:", rfc_accuracy)

# 输出随机森林分类器的混淆矩阵
rfc_cm = confusion_matrix(Y_test, rfc_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rfc_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()

# XGBoost分类器
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)
xgb_pred = xgb.predict(X_test)
xgb_accuracy = accuracy_score(Y_test, xgb_pred)
print("XGBoost Classifier Accuracy:", xgb_accuracy)

# 输出XGBoost分类器的混淆矩阵
xgb_cm = confusion_matrix(Y_test, xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - XGBoost Classifier')
plt.show()

# 决策树分类器 (Gini)
dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train, Y_train)
dt_gini_pred = dt_gini.predict(X_test)
dt_gini_accuracy = accuracy_score(Y_test, dt_gini_pred)
print("Decision Tree Classifier (Gini) Accuracy:", dt_gini_accuracy)

# 输出决策树分类器 (Gini) 的混淆矩阵
dt_gini_cm = confusion_matrix(Y_test, dt_gini_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_gini_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Decision Tree Classifier (Gini)')
plt.show()

# 决策树分类器 (Entropy)
dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train, Y_train)
dt_entropy_pred = dt_entropy.predict(X_test)
dt_entropy_accuracy = accuracy_score(Y_test, dt_entropy_pred)
print("Decision Tree Classifier (Entropy) Accuracy:", dt_entropy_accuracy)

# 输出决策树分类器 (Entropy) 的混淆矩阵
dt_entropy_cm = confusion_matrix(Y_test, dt_entropy_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_entropy_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Decision Tree Classifier (Entropy)')
plt.show()

# 神经网络分类器
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 将标签转换为numpy数组
Y_train_nn = np.array(Y_train)
Y_test_nn = np.array(Y_test)

# 训练神经网络模型
model.fit(X_train, Y_train_nn, batch_size=64, epochs=40, verbose=1)

# 在测试集上评估模型
nn_pred = model.predict(X_test)
nn_pred = np.round(nn_pred).flatten()
nn_accuracy = accuracy_score(Y_test_nn, nn_pred)
print("Neural Network Classifier Accuracy:", nn_accuracy)

# 输出神经网络分类器的混淆矩阵
nn_cm = confusion_matrix(Y_test_nn, nn_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Neural Network Classifier')
plt.show()

# 准确度比较表
accuracy_df = pd.DataFrame({
    'Classifier': ['Random Forest', 'XGBoost', 'Decision Tree (Gini)', 'Decision Tree (Entropy)', 'Neural Network'],
    'Accuracy': [rfc_accuracy, xgb_accuracy, dt_gini_accuracy, dt_entropy_accuracy, nn_accuracy]
})

# 使用柱状图表示准确度比较表
plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Accuracy', data=accuracy_df)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0.8, 1.0)
plt.show()
