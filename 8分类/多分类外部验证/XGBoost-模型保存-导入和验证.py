import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import joblib  # 用于保存模型

# 读取数据
df = pd.read_csv('E:/RS/sm/ALL-Feature/XGBoost/final/XGBoost-16.csv')

# 划分特征和目标变量
X = df.drop(['label'], axis=1)
y = df['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['label'])

# 创建并训练XGBoost分类器
rf_classifier = xgb.XGBClassifier(colsample_bytree=0.5, learning_rate=0.2, max_depth=3, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# 生成分类报告
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7), dpi=1200)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion matrix heat map', fontsize=15)
plt.show()

# 计算ROC曲线和AUC
ytest_proba_rf = rf_classifier.predict_proba(X_test)
ytest_one_rf = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7])

rf_AUC = {}
rf_FPR = {}
rf_TPR = {}

for i in range(ytest_one_rf.shape[1]):
    rf_FPR[i], rf_TPR[i], _ = metrics.roc_curve(ytest_one_rf[:, i], ytest_proba_rf[:, i])
    rf_AUC[i] = metrics.auc(rf_FPR[i], rf_TPR[i])

# 计算宏平均ROC曲线和AUC
rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(ytest_one_rf.shape[1])]))
rf_TPR_all = np.zeros_like(rf_FPR_final)

for i in range(ytest_one_rf.shape[1]):
    rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]

rf_AUC_final = metrics.auc(rf_FPR_final, rf_TPR_final)
print(f"Macro Average AUC with XGBoost: {rf_AUC_final}")

# 保存模型
joblib.dump(rf_classifier, 'E:/RS/sm/ALL-Feature/XGBoost/final/XGBoost-16.pkl')
print("模型已保存到本地")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import joblib  # 用于加载模型
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,precision_recall_curve,auc

# 加载已保存的模型
rf_classifier = joblib.load('E:/RS/sm/ALL-Feature/XGBoost/final/XGBoost-16.pkl')
print("模型已成功加载")

# 读取外部数据
data = pd.read_csv('E:/RS/sm/ALL-Feature/XGBoost/final/YAN1.csv')

# 使用外部数据进行预测
predictions = rf_classifier.predict(data.drop(['label'], axis=1))

# 生成分类报告
print(classification_report(data['label'], predictions))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(data['label'], predictions)
plt.figure(figsize=(10, 7), dpi=300)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':4},
            fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('Confusion matrix heat map', fontsize=8)
plt.show()


# 计算各类指标: 准确性、敏感性、特异性、PPV、NPV、F1值，AUC
def calculate_metrics(conf_matrix, class_idx, y_true_bin, y_proba):
    TP = conf_matrix[class_idx, class_idx]
    FP = conf_matrix[:, class_idx].sum() - TP
    FN = conf_matrix[class_idx, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # 召回率/敏感性
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # 特异性
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0  # 阳性预测值
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 0  # 阴性预测值
    F1 = 2 * (PPV * sensitivity) / (PPV + sensitivity) if (PPV + sensitivity) != 0 else 0  # F1值

    # 计算 AUC
    auc = roc_auc_score(y_true_bin[:, class_idx], y_proba[:, class_idx])

    return accuracy, sensitivity, specificity, PPV, NPV, F1, auc


# 对标签进行one-hot编码
ytest_proba_rf = rf_classifier.predict_proba(data.drop(['label'], axis=1))
ytest_one_rf = label_binarize(data['label'], classes=[0, 1, 2, 3, 4, 5, 6, 7])

# 计算每个类别的指标
for i in range(conf_matrix.shape[0]):
    accuracy, sensitivity, specificity, PPV, NPV, F1, auc = calculate_metrics(conf_matrix, i, ytest_one_rf,
                                                                              ytest_proba_rf)
    print(f"Class {i + 1} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Positive Predictive Value (PPV): {PPV:.4f}")
    print(f"  Negative Predictive Value (NPV): {NPV:.4f}")
    print(f"  F1 Score: {F1:.4f}")
    print(f"  AUC: {auc:.4f}\n")

# 计算ROC曲线和AUC
rf_AUC = {}
rf_FPR = {}
rf_TPR = {}

for i in range(ytest_one_rf.shape[1]):
    rf_FPR[i], rf_TPR[i], _ = metrics.roc_curve(ytest_one_rf[:, i], ytest_proba_rf[:, i])
    rf_AUC[i] = metrics.auc(rf_FPR[i], rf_TPR[i])

# 计算宏平均ROC曲线和AUC
rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(ytest_one_rf.shape[1])]))
rf_TPR_all = np.zeros_like(rf_FPR_final)

for i in range(ytest_one_rf.shape[1]):
    rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]

rf_AUC_final = metrics.auc(rf_FPR_final, rf_TPR_final)
print(f"Macro Average AUC with External Data: {rf_AUC_final}")

# 绘制ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
# 使用不同的颜色和线型
plt.plot(rf_FPR[0], rf_TPR[0], color='b', linestyle='-', label='Class 1 ROC  AUC={:.4f}'.format(rf_AUC[0]), lw=0.8)
plt.plot(rf_FPR[1], rf_TPR[1], color='g', linestyle='-', label='Class 2 ROC  AUC={:.4f}'.format(rf_AUC[1]), lw=0.8)
plt.plot(rf_FPR[2], rf_TPR[2], color='r', linestyle='-', label='Class 3 ROC  AUC={:.4f}'.format(rf_AUC[2]), lw=0.8)
plt.plot(rf_FPR[3], rf_TPR[3], color='c', linestyle='-', label='Class 4 ROC  AUC={:.4f}'.format(rf_AUC[3]), lw=0.8)
plt.plot(rf_FPR[4], rf_TPR[4], color='m', linestyle='-', label='Class 5 ROC  AUC={:.4f}'.format(rf_AUC[4]), lw=0.8)
plt.plot(rf_FPR[5], rf_TPR[5], color='y', linestyle='-', label='Class 6 ROC  AUC={:.4f}'.format(rf_AUC[5]), lw=0.8)
plt.plot(rf_FPR[6], rf_TPR[6], color='#FFA500', linestyle='-', label='Class 7 ROC  AUC={:.4f}'.format(rf_AUC[6]), lw=0.8)
plt.plot(rf_FPR[7], rf_TPR[7], color='#9467bd', linestyle='-', label='Class 8 ROC  AUC={:.4f}'.format(rf_AUC[7]), lw=0.8)
# 宏平均ROC曲线
plt.plot(rf_FPR_final, rf_TPR_final, color='#000000', linestyle='-', label='Macro Average ROC  AUC={:.4f}'.format(rf_AUC_final), lw=1)
# 45度参考线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('XGBoost Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

from sklearn.metrics import precision_recall_curve, auc

pr_precision = {}
pr_recall = {}
pr_auc = {}

# For each class, calculate PR curve and AUC
for i in range(ytest_one_rf.shape[1]):
    pr_precision[i], pr_recall[i], _ = precision_recall_curve(ytest_one_rf[:, i], ytest_proba_rf[:, i])
    pr_auc[i] = auc(pr_recall[i], pr_precision[i])

# Plot PR curves
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(pr_recall[0], pr_precision[0], color='b', linestyle='-',label=f'Class1 PR  AUC={pr_auc[0]:.4f}', lw=0.8)
plt.plot(pr_recall[1], pr_precision[1], color='g', linestyle='-',label=f'Class2 PR  AUC={pr_auc[1]:.4f}', lw=0.8)
plt.plot(pr_recall[2], pr_precision[2], color='r', linestyle='-',label=f'Class3 PR  AUC={pr_auc[2]:.4f}', lw=0.8)
plt.plot(pr_recall[3], pr_precision[3], color='c', linestyle='-',label=f'Class4 PR  AUC={pr_auc[3]:.4f}', lw=0.8)
plt.plot(pr_recall[4], pr_precision[4], color='m', linestyle='-',label=f'Class5 PR  AUC={pr_auc[4]:.4f}', lw=0.8)
plt.plot(pr_recall[5], pr_precision[5], color='y', linestyle='-',label=f'Class6 PR  AUC={pr_auc[5]:.4f}', lw=0.8)
plt.plot(pr_recall[6], pr_precision[6], color='k', linestyle='-',label=f'Class7 PR  AUC={pr_auc[6]:.4f}', lw=0.8)
plt.plot(pr_recall[7], pr_precision[7], color='#9467bd', linestyle='-',label=f'Class8 PR  AUC={pr_auc[7]:.4f}', lw=0.8)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('XGBoost Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', framealpha=0.9, fontsize=5)
plt.show()


