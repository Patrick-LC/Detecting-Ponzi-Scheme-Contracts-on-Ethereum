import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.impute import SimpleImputer
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_other_models():
    # 读取数据
    data = pd.read_csv('/Users/lc/Graduation Thesis/etherscan/transaction_dataset.csv')

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
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
    X = imputer.fit_transform(data.drop('FLAG', axis=1))
    Y = data['FLAG']

    scaler = StandardScaler()  # 数据标准化
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 随机森林分类器
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    rfc_pred = rfc.predict(X_test)
    rfc_accuracy = accuracy_score(Y_test, rfc_pred)
    rfc_precision = precision_score(Y_test, rfc_pred)
    rfc_recall = recall_score(Y_test, rfc_pred)
    rfc_f1_score = f1_score(Y_test, rfc_pred)

    # XGBoost分类器
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_accuracy = accuracy_score(Y_test, xgb_pred)
    xgb_precision = precision_score(Y_test, xgb_pred)
    xgb_recall = recall_score(Y_test, xgb_pred)
    xgb_f1_score = f1_score(Y_test, xgb_pred)

    # 决策树分类器 (Gini)
    dt_gini = DecisionTreeClassifier(criterion='gini')
    dt_gini.fit(X_train, Y_train)
    dt_gini_pred = dt_gini.predict(X_test)
    dt_gini_accuracy = accuracy_score(Y_test, dt_gini_pred)
    dt_gini_precision = precision_score(Y_test, dt_gini_pred)
    dt_gini_recall = recall_score(Y_test, dt_gini_pred)
    dt_gini_f1_score = f1_score(Y_test, dt_gini_pred)

    # 决策树分类器 (Entropy)
    dt_entropy = DecisionTreeClassifier(criterion='entropy')
    dt_entropy.fit(X_train, Y_train)
    dt_entropy_pred = dt_entropy.predict(X_test)
    dt_entropy_accuracy = accuracy_score(Y_test, dt_entropy_pred)
    dt_entropy_precision = precision_score(Y_test, dt_entropy_pred)
    dt_entropy_recall = recall_score(Y_test, dt_entropy_pred)
    dt_entropy_f1_score = f1_score(Y_test, dt_entropy_pred)

    # 神经网络分类器
    model = Sequential()
    model.add(Dense(45, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
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
    history = model.fit(X_train, Y_train_nn, batch_size=64, epochs=150, verbose=1,
                        validation_data=(X_test, Y_test_nn),
                        callbacks=[early_stopping, model_checkpoint]
                        )

    # 加载在验证集上性能最好的模型
    model.load_weights('best_model.h5')

    # 在测试集上评估模型
    nn_pred = model.predict(X_test)
    nn_pred = np.round(nn_pred).flatten()
    nn_accuracy = accuracy_score(Y_test_nn, nn_pred)
    nn_precision = precision_score(Y_test_nn, nn_pred)
    nn_recall = recall_score(Y_test_nn, nn_pred)
    nn_f1_score = f1_score(Y_test_nn, nn_pred)

    return {
        "Random Forest Classifier": {
            "Accuracy": rfc_accuracy,
            "Precision": rfc_precision,
            "Recall": rfc_recall,
            "F1 Score": rfc_f1_score
        },
        "XGBoost Classifier": {
            "Accuracy": xgb_accuracy,
            "Precision": xgb_precision,
            "Recall": xgb_recall,
            "F1 Score": xgb_f1_score
        },
        "Decision Tree Classifier (Gini)": {
            "Accuracy": dt_gini_accuracy,
            "Precision": dt_gini_precision,
            "Recall": dt_gini_recall,
            "F1 Score": dt_gini_f1_score
        },
        "Decision Tree Classifier (Entropy)": {
            "Accuracy": dt_entropy_accuracy,
            "Precision": dt_entropy_precision,
            "Recall": dt_entropy_recall,
            "F1 Score": dt_entropy_f1_score
        },
        "Neural Network Classifier": {
            "Accuracy": nn_accuracy,
            "Precision": nn_precision,
            "Recall": nn_recall,
            "F1 Score": nn_f1_score
        }
    }

if __name__ == "__main__":
    results = train_and_evaluate_other_models()
    print(results)
