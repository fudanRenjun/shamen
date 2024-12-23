import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.metrics import precision_recall_curve

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载保存的模型
model_path = 'E:/RS/sm/ALL-Feature/MLP/MLP-16.pkl'
mlp_model_loaded = joblib.load(model_path)
print("模型已加载")

# 加载新的预测数据
data_to_predict = pd.read_csv('E:/RS/sm/ALL-Feature/MLP/YAN-W1.csv')

# 假设第一列是标签列，其他列是特征
X_new = data_to_predict.iloc[:, 1:]  # 特征列
y_true = data_to_predict.iloc[:, 0]  # 真实标签（如果有的话）

# 检查新数据的形状
print("新数据的特征形状:", X_new.shape)
print("新数据的真实标签形状:", y_true.shape)

# 加载之前拟合的标准化器
scaler = joblib.load('E:/RS/sm/ALL-Feature/MLP/scaler.pkl')  # 确保路径正确

# 使用之前拟合的标准化器转换新数据
X_new_scaled = scaler.transform(X_new)  # 只进行转换
print("标准化后的新数据特征形状:", X_new_scaled.shape)

# 使用加载的模型进行预测
y_pred_new = mlp_model_loaded.predict(X_new_scaled)
y_pred_proba = mlp_model_loaded.predict_proba(X_new_scaled)

# 输出预测结果
print("新数据的预测结果:")
print(y_pred_new)

# 如果有真实标签，计算混淆矩阵和分类报告
if 'y_true' in locals():  # 检查 y_true 是否存在
    print("Classification Report for New Data:")
    print(classification_report(y_true, y_pred_new))

    # 计算混淆矩阵
    conf_matrix_new = confusion_matrix(y_true, y_pred_new)

    # 计算指标
    TP = np.diag(conf_matrix_new)  # 真阳性
    FP = np.sum(conf_matrix_new, axis=0) - TP  # 假阳性
    FN = np.sum(conf_matrix_new, axis=1) - TP  # 假阴性
    TN = np.sum(conf_matrix_new) - (FP + FN + TP)  # 真阴性

    # 准确性
    accuracy = np.sum(TP) / np.sum(conf_matrix_new)

    # 敏感性（召回率）
    sensitivity = TP / (TP + FN)

    # 特异性
    specificity = TN / (TN + FP)

    # 阳性预测值（PPV）
    PPV = TP / (TP + FP)

    # 阴性预测值（NPV）
    NPV = TN / (TN + FN)

    # F1 值
    f1 = f1_score(y_true, y_pred_new, average='weighted')

    # 计算 ROC 和 AUC（多分类情况，需要进行 One-vs-Rest 处理）
    mlp_FPR = []
    mlp_TPR = []
    mlp_AUC = []

    for i in range(8):  # 0-7 标签
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        mlp_FPR.append(fpr)
        mlp_TPR.append(tpr)
        mlp_AUC.append(auc(fpr, tpr))

    # 计算宏平均ROC
    mlp_FPR_final = np.linspace(0, 1, 100)
    mlp_TPR_final = np.zeros_like(mlp_FPR_final)

    for i in range(len(mlp_FPR)):
        mlp_TPR_final += np.interp(mlp_FPR_final, mlp_FPR[i], mlp_TPR[i])

    mlp_TPR_final /= len(mlp_FPR)
    mlp_AUC_final = auc(mlp_FPR_final, mlp_TPR_final)

    # 输出计算的指标
    print("准确性:", accuracy)
    print("敏感性:", sensitivity)
    print("特异性:", specificity)
    print("阳性预测值 (PPV):", PPV)
    print("阴性预测值 (NPV):", NPV)
    print("F1 值:", f1)
    print("每个类的 ROC AUC:", mlp_AUC)
    print("宏平均 ROC AUC:", mlp_AUC_final)

    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(conf_matrix_new, annot=True, annot_kws={'size': 4},
                fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('Predicted Label', fontsize=7)
    plt.ylabel('True Label', fontsize=7)
    plt.title('Confusion Matrix Heat Map for New Data', fontsize=8)
    plt.show()

    # 绘制ROC曲线
    plt.figure(figsize=(10, 5), dpi=300)
    # 使用不同的颜色和线型绘制每个类别的ROC曲线
    plt.plot(mlp_FPR[0], mlp_TPR[0], color='b', linestyle='-', label='Class 1 ROC  AUC={:.4f}'.format(mlp_AUC[0]),
             lw=0.8)
    plt.plot(mlp_FPR[1], mlp_TPR[1], color='g', linestyle='-', label='Class 2 ROC  AUC={:.4f}'.format(mlp_AUC[1]),
             lw=0.8)
    plt.plot(mlp_FPR[2], mlp_TPR[2], color='r', linestyle='-', label='Class 3 ROC  AUC={:.4f}'.format(mlp_AUC[2]),
             lw=0.8)
    plt.plot(mlp_FPR[3], mlp_TPR[3], color='c', linestyle='-', label='Class 4 ROC  AUC={:.4f}'.format(mlp_AUC[3]),
             lw=0.8)
    plt.plot(mlp_FPR[4], mlp_TPR[4], color='m', linestyle='-', label='Class 5 ROC  AUC={:.4f}'.format(mlp_AUC[4]),
             lw=0.8)
    plt.plot(mlp_FPR[5], mlp_TPR[5], color='y', linestyle='-', label='Class 6 ROC  AUC={:.4f}'.format(mlp_AUC[5]),
             lw=0.8)
    plt.plot(mlp_FPR[6], mlp_TPR[6], color='#FFA500', linestyle='-', label='Class 7 ROC  AUC={:.4f}'.format(mlp_AUC[6]),
             lw=0.8)
    plt.plot(mlp_FPR[7], mlp_TPR[7], color='#9467bd', linestyle='-', label='Class 8 ROC  AUC={:.4f}'.format(mlp_AUC[7]),
             lw=0.8)

    # 绘制宏平均ROC曲线
    plt.plot(mlp_FPR_final, mlp_TPR_final, color='#000000', linestyle='-',
             label='Macro Average ROC  AUC={:.4f}'.format(mlp_AUC_final), lw=1)

    # 添加45度参考线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')

    # 设置坐标轴和标题
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('False Positive Rate (FPR)', fontsize=7)
    plt.ylabel('True Positive Rate (TPR)', fontsize=7)
    plt.title('MLP Classification ROC Curves and AUC', fontsize=8)
    plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
    plt.show()

    # 计算 PR 和 AUC
    pr_Recall = []
    pr_Precision = []
    pr_AUC = []

    for i in range(8):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_pred_proba[:, i])
        pr_Recall.append(recall)
        pr_Precision.append(precision)
        pr_AUC.append(auc(recall, precision))

    # 绘制PR曲线
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(pr_Recall[0], pr_Precision[0], color='b', linestyle='-', label=f'Class 1 PR  AUC={pr_AUC[0]:.4f}', lw=0.8)
    plt.plot(pr_Recall[1], pr_Precision[1], color='g', linestyle='-', label=f'Class 2 PR  AUC={pr_AUC[1]:.4f}', lw=0.8)
    plt.plot(pr_Recall[2], pr_Precision[2], color='r', linestyle='-', label=f'Class 3 PR  AUC={pr_AUC[2]:.4f}', lw=0.8)
    plt.plot(pr_Recall[3], pr_Precision[3], color='c', linestyle='-', label=f'Class 4 PR  AUC={pr_AUC[3]:.4f}', lw=0.8)
    plt.plot(pr_Recall[4], pr_Precision[4], color='m', linestyle='-', label=f'Class 5 PR  AUC={pr_AUC[4]:.4f}', lw=0.8)
    plt.plot(pr_Recall[5], pr_Precision[5], color='y', linestyle='-', label=f'Class 6 PR  AUC={pr_AUC[5]:.4f}', lw=0.8)
    plt.plot(pr_Recall[6], pr_Precision[6], color='#FFA500', linestyle='-', label=f'Class 7 PR  AUC={pr_AUC[6]:.4f}',
             lw=0.8)
    plt.plot(pr_Recall[7], pr_Precision[7], color='#9467bd', linestyle='-', label=f'Class 8 PR  AUC={pr_AUC[7]:.4f}',
             lw=0.8)

    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('Recall', fontsize=7)
    plt.ylabel('Precision', fontsize=7)
    plt.title('MLP Classification PR Curves and AUC', fontsize=8)
    plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
    plt.show()
else:
    print("未提供新数据的真实标签，无法计算混淆矩阵和")