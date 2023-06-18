import matplotlib.pyplot as plt
import seaborn as sns
from cnn_model import train_and_evaluate_cnn
from final import train_and_evaluate_other_models

# 执行CNN模型训练和评估
cnn_results = train_and_evaluate_cnn()

# 执行其他模型训练和评估
other_models_results = train_and_evaluate_other_models()

# 提取性能指标
models = ['CNN', 'RF', 'XGB', 'DT-Gini', 'DT-E', 'ANN']
accuracy = [cnn_results['Accuracy'], other_models_results['Random Forest Classifier']['Accuracy'],
            other_models_results['XGBoost Classifier']['Accuracy'],
            other_models_results['Decision Tree Classifier (Gini)']['Accuracy'],
            other_models_results['Decision Tree Classifier (Entropy)']['Accuracy'],
            other_models_results['Neural Network Classifier']['Accuracy']]
precision = [None, other_models_results['Random Forest Classifier']['Precision'],
             other_models_results['XGBoost Classifier']['Precision'],
             other_models_results['Decision Tree Classifier (Gini)']['Precision'],
             other_models_results['Decision Tree Classifier (Entropy)']['Precision'],
             other_models_results['Neural Network Classifier']['Precision']]
recall = [None, other_models_results['Random Forest Classifier']['Recall'],
          other_models_results['XGBoost Classifier']['Recall'],
          other_models_results['Decision Tree Classifier (Gini)']['Recall'],
          other_models_results['Decision Tree Classifier (Entropy)']['Recall'],
          other_models_results['Neural Network Classifier']['Recall']]
f1_score = [cnn_results['F1 Score'], other_models_results['Random Forest Classifier']['F1 Score'],
            other_models_results['XGBoost Classifier']['F1 Score'],
            other_models_results['Decision Tree Classifier (Gini)']['F1 Score'],
            other_models_results['Decision Tree Classifier (Entropy)']['F1 Score'],
            other_models_results['Neural Network Classifier']['F1 Score']]

# 设置图表大小
plt.figure(figsize=(10, 8))

# 绘制Accuracy折线图
plt.subplot(221)
plt.plot(models, accuracy, marker='o')
plt.title('Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)

# 绘制Precision折线图
plt.subplot(222)
plt.plot(models[1:], precision[1:], marker='o')
plt.title('Precision Comparison')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.ylim(0.8, 1.0)

# 绘制Recall折线图
plt.subplot(223)
plt.plot(models[1:], recall[1:], marker='o')
plt.title('Recall Comparison')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.ylim(0.8, 1.0)

# 绘制F1 Score折线图
plt.subplot(224)
plt.plot(models, f1_score, marker='o')
plt.title('F1 Score Comparison')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.ylim(0.8, 1.0)

# 调整子图间距
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# 显示图表
plt.show()
