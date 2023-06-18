import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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

# 定义隐藏层神经元个数的范围
neuron_nums = [8, 16, 32, 64, 128]

# 初始化结果列表
losses = []
accuracies = []

for num_neurons in neuron_nums:
    # 神经网络分类器
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 将标签转换为numpy数组
    Y_train_nn = np.array(Y_train)
    Y_test_nn = np.array(Y_test)

    # 定义EarlyStopping和ModelCheckpoint回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # 训练神经网络模型
    history = model.fit(X_train, Y_train_nn, batch_size=64, epochs=160, verbose=1,
                        validation_data=(X_test, Y_test_nn),
                        callbacks=[early_stopping, model_checkpoint])

    # 加载在验证集上性能最好的模型
    model.load_weights('best_model.h5')

    # 在测试集上评估模型
    nn_pred = model.predict(X_test)
    nn_pred = np.round(nn_pred).flatten()
    nn_f1_score = f1_score(Y_test_nn, nn_pred)
    print("Neural Network Classifier F1 Score ({} neurons):".format(num_neurons), nn_f1_score)

    # 根据混淆矩阵计算精度
    tn, fp, fn, tp = confusion_matrix(Y_test_nn, nn_pred).ravel()
    nn_accuracy = (tn + tp + fn) / (tn + fp + tp)
    print("Neural Network Classifier Accuracy ({} neurons):".format(num_neurons), nn_accuracy)

    # 保存结果
    losses.append(history.history['loss'][-1])
    accuracies.append(nn_accuracy)

# 绘制损失-神经元个数柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(len(neuron_nums)), losses, align='center')
plt.xlabel('Neuron Number')
plt.ylabel('Loss')
plt.title('Loss vs. Neuron Number')
plt.xticks(range(len(neuron_nums)), neuron_nums)
plt.show()

# 绘制准确率-神经元个数柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(len(neuron_nums)), accuracies, align='center')
plt.xlabel('Neuron Number')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Neuron Number')
plt.xticks(range(len(neuron_nums)), neuron_nums)
plt.show()

