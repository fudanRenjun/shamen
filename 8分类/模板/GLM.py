import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import statsmodels.api as sm

# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载CSV文件
data = pd.read_csv('E:/RS/sm/XUN.csv')

print(data.isnull().sum())
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 划分训练集和测试集
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = MinMaxScaler()
xtrain_s = scaler.fit_transform(xtrain)
xtest_s = scaler.transform(xtest)

# 添加常数项（截距项），因为GLM需要明确指定截距项
xtrain_s = sm.add_constant(xtrain_s)
xtest_s = sm.add_constant(xtest_s)

# 使用多项式逻辑回归（MNLogit）模型进行多分类
model_mnlogit = sm.MNLogit(ytrain, xtrain_s)

# 训练模型
result = model_mnlogit.fit()

# 预测测试集
y_pred_proba = result.predict(xtest_s)

# 将预测概率转换为类别
y_pred_class = np.argmax(y_pred_proba, axis=1)

# 打印分类报告
print("Classification Report:")
print(classification_report(ytest, y_pred_class))

# 计算混淆矩阵
conf_matrix = confusion_matrix(ytest, y_pred_class)

# 绘制热力图
plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(conf_matrix, annot=True, annot_kws={'size':4},
            fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Predicted Label', fontsize=7)
plt.ylabel('True Label', fontsize=7)
plt.title('Confusion matrix heat map', fontsize=8)
plt.show()

# 计算并打印每个类别的指标
num_classes = conf_matrix.shape[0]
for i in range(num_classes):
    tp = conf_matrix[i, i]  # True Positive for class i
    fp = conf_matrix[:, i].sum() - tp  # False Positive for class i
    fn = conf_matrix[i, :].sum() - tp  # False Negative for class i
    tn = conf_matrix.sum() - (tp + fp + fn)  # True Negative for class i

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) != 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) != 0 else 0
    sensitivity = recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    print(f"Class {i + 1}:")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value: {positive_predictive_value:.4f}")
    print(f"Negative Predictive Value: {negative_predictive_value:.4f}")
    print(f"Sensitivity (Recall for class {i + 1}): {sensitivity:.4f}")

# 计算ROC曲线和AUC
ytest_one_mnlogit = label_binarize(ytest, classes=np.unique(y))
mnlogit_AUC = {}
mnlogit_FPR = {}
mnlogit_TPR = {}

for i in range(ytest_one_mnlogit.shape[1]):
    mnlogit_FPR[i], mnlogit_TPR[i], _ = roc_curve(ytest_one_mnlogit[:, i], y_pred_proba[:, i])
    mnlogit_AUC[i] = auc(mnlogit_FPR[i], mnlogit_TPR[i])
print("ROC AUC for each class:", mnlogit_AUC)

# 计算宏平均ROC曲线和AUC
mnlogit_FPR_final = np.unique(np.concatenate([mnlogit_FPR[i] for i in range(ytest_one_mnlogit.shape[1])]))
mnlogit_TPR_all = np.zeros_like(mnlogit_FPR_final)
for i in range(ytest_one_mnlogit.shape[1]):
    mnlogit_TPR_all += np.interp(mnlogit_FPR_final, mnlogit_FPR[i], mnlogit_TPR[i])
mnlogit_TPR_final = mnlogit_TPR_all / ytest_one_mnlogit.shape[1]
mnlogit_AUC_final = auc(mnlogit_FPR_final, mnlogit_TPR_final)
print(f"Macro Average AUC with MNLogit: {mnlogit_AUC_final}")

# 绘制ROC曲线
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(mnlogit_FPR[0], mnlogit_TPR[0], color='b', linestyle='-', label='Class 1 ROC  AUC={:.4f}'.format(mnlogit_AUC[0]), lw=0.8)
plt.plot(mnlogit_FPR[1], mnlogit_TPR[1], color='g', linestyle='-', label='Class 2 ROC  AUC={:.4f}'.format(mnlogit_AUC[1]), lw=0.8)
plt.plot(mnlogit_FPR[2], mnlogit_TPR[2], color='r', linestyle='-', label='Class 3 ROC  AUC={:.4f}'.format(mnlogit_AUC[2]), lw=0.8)
plt.plot(mnlogit_FPR[3], mnlogit_TPR[3], color='c', linestyle='-', label='Class 4 ROC  AUC={:.4f}'.format(mnlogit_AUC[3]), lw=0.8)
plt.plot(mnlogit_FPR[4], mnlogit_TPR[4], color='m', linestyle='-', label='Class 5 ROC  AUC={:.4f}'.format(mnlogit_AUC[4]), lw=0.8)
plt.plot(mnlogit_FPR[5], mnlogit_TPR[5], color='y', linestyle='-', label='Class 6 ROC  AUC={:.4f}'.format(mnlogit_AUC[5]), lw=0.8)
plt.plot(mnlogit_FPR[6], mnlogit_TPR[6], color='#FFA500', linestyle='-', label='Class 7 ROC  AUC={:.4f}'.format(mnlogit_AUC[6]), lw=0.8)
plt.plot(mnlogit_FPR[7], mnlogit_TPR[7], color='#9467bd', linestyle='-', label='Class 8 ROC  AUC={:.4f}'.format(mnlogit_AUC[7]), lw=0.8)
# 宏平均ROC曲线
plt.plot(mnlogit_FPR_final, mnlogit_TPR_final, color='#000000', linestyle='-', label='Macro Average ROC  AUC={:.4f}'.format(mnlogit_AUC_final), lw=1)
# 45度参考线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='45 Degree Reference Line')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('False Positive Rate (FPR)', fontsize=7)
plt.ylabel('True Positive Rate (TPR)', fontsize=7)
plt.title('MNLogit Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()