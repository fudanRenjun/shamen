import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
import seaborn as sns


# 显示中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 加载CSV文件
data = pd.read_csv('E:/RS/sm/XUN1.csv')

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


# 使用随机森林建模
model_rf = GradientBoostingClassifier(
                                   max_depth= 5,
                                   min_samples_leaf= 10,
                                   min_samples_split= 5,
                                   n_estimators=50,
                                   )

#训练模型
model_rf.fit(xtrain_s, ytrain)

# 预测测试集
y_pred = model_rf.predict(xtest_s)
y_pred_proba = model_rf.predict_proba(xtest_s)

# 打印分类报告
print("Classification Report:")
print(classification_report(ytest, y_pred))

# 计算混淆矩阵
conf_matrix = confusion_matrix(ytest, y_pred)
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
ytest_one_rf = label_binarize(ytest, classes=np.unique(y))
rf_AUC = {}
rf_FPR = {}
rf_TPR = {}

for i in range(ytest_one_rf.shape[1]):
    rf_FPR[i], rf_TPR[i], _ = roc_curve(ytest_one_rf[:, i], y_pred_proba[:, i])
    rf_AUC[i] = auc(rf_FPR[i], rf_TPR[i])
print("ROC AUC for each class:", rf_AUC)

# 计算宏平均ROC曲线和AUC
rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(ytest_one_rf.shape[1])]))
rf_TPR_all = np.zeros_like(rf_FPR_final)
for i in range(ytest_one_rf.shape[1]):
    rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]
rf_AUC_final = auc(rf_FPR_final, rf_TPR_final)
print(f"Macro Average AUC with GradientBoosting: {rf_AUC_final}")

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
plt.title('GradientBoosting Classification ROC Curves and AUC', fontsize=8)
plt.legend(loc='lower right', framealpha=0.9, fontsize=5)
plt.show()

# 计算PR曲线和平均精确度
pr_AUC = {}
pr_Precision = {}
pr_Recall = {}

for i in range(ytest_one_rf.shape[1]):
    pr_Recall[i], pr_Precision[i], _ = precision_recall_curve(ytest_one_rf[:, i], y_pred_proba[:, i])
    pr_AUC[i] = average_precision_score(ytest_one_rf[:, i], y_pred_proba[:, i])
print("PR AUC for each class:", pr_AUC)

# 绘制PR曲线
plt.figure(figsize=(10, 5), dpi=300)
# 使用不同的颜色和线型
plt.plot(pr_Recall[0], pr_Precision[0], color='b', linestyle='-',label=f'Class1 PR  AUC={pr_AUC[0]:.4f}', lw=0.8)
plt.plot(pr_Recall[1], pr_Precision[1], color='g', linestyle='-',label=f'Class2 PR  AUC={pr_AUC[1]:.4f}', lw=0.8)
plt.plot(pr_Recall[2], pr_Precision[2], color='r', linestyle='-',label=f'Class3 PR  AUC={pr_AUC[2]:.4f}', lw=0.8)
plt.plot(pr_Recall[3], pr_Precision[3], color='c', linestyle='-',label=f'Class4 PR  AUC={pr_AUC[3]:.4f}', lw=0.8)
plt.plot(pr_Recall[4], pr_Precision[4], color='m', linestyle='-',label=f'Class5 PR  AUC={pr_AUC[4]:.4f}', lw=0.8)
plt.plot(pr_Recall[5], pr_Precision[5], color='y', linestyle='-',label=f'Class6 PR  AUC={pr_AUC[5]:.4f}', lw=0.8)
plt.plot(pr_Recall[6], pr_Precision[6], color='k', linestyle='-',label=f'Class7 PR  AUC={pr_AUC[6]:.4f}', lw=0.8)
plt.plot(pr_Recall[7], pr_Precision[7], color='#9467bd', linestyle='-',label=f'Class8 PR  AUC={pr_AUC[7]:.4f}', lw=0.8)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Recall', fontsize=7)
plt.ylabel('Precision', fontsize=7)
plt.title('GradientBoosting Classification PR Curves and AUC', fontsize=8)
plt.legend(loc='lower left', framealpha=0.9, fontsize=5)
plt.show()

# 计算特征重要性并进行排序
feature_importances = model_rf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]  # 从大到小排序

# 7. 绘制前20个特征的重要性直方图
top_20_indices = sorted_indices[:20]
top_20_feature_importances = feature_importances[top_20_indices]
top_20_feature_names = X.columns[top_20_indices]

plt.figure(figsize=(12, 8), dpi=300)
plt.barh(top_20_feature_names, top_20_feature_importances, color='skyblue')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Importance', fontsize=7)
plt.ylabel('Feature', fontsize=7)
plt.title('Top 20 Feature Importances', fontsize=8)
plt.gca().invert_yaxis()  # 反转Y轴，使得最重要的特征在上面
plt.show()

# 初始化保存结果的列表
feature_counts = []
roc_aucs = []

# 4. 逐步增加特征数量进行模型训练和评估
for num_features in range(1, 21):
    # 选择最重要的num_features个特征
    top_indices = sorted_indices[:num_features]
    xtrain_subset = xtrain_s[:, top_indices]
    xtest_subset = xtest_s[:, top_indices]

    # 使用随机森林进行建模
    model_rf = GradientBoostingClassifier(learning_rate= 0.1,
                                   max_depth= 7,
                                   max_features= 'log2',
                                   min_samples_leaf= 1,
                                   min_samples_split= 2,
                                   n_estimators=50,
                                   subsample=1.0)
    model_rf.fit(xtrain_subset, ytrain)

    # 预测并计算ROC AUC
    y_pred_proba = model_rf.predict_proba(xtest_subset)
    ytest_one_hot = pd.get_dummies(ytest).values
    auc_values = []
    for i in range(ytest_one_hot.shape[1]):
        auc_value = roc_auc_score(ytest_one_hot[:, i], y_pred_proba[:, i])
        auc_values.append(auc_value)

    # 计算宏平均ROC AUC
    average_auc = np.mean(auc_values)

    # 记录特征数量和对应的ROC AUC值
    feature_counts.append(num_features)
    roc_aucs.append(average_auc)

    print(f"Features: {num_features}, ROC AUC: {average_auc:.4f}")

# 5. 绘制ROC AUC折线图
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(feature_counts, roc_aucs, marker='o', linestyle='-', color='b')
plt.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('Number of Features', fontsize=7)
plt.ylabel('ROC AUC', fontsize=7)
plt.title('ROC AUC vs. Number of Features', fontsize=8)
plt.xticks(np.arange(1, 21, step=1))
plt.grid(True)
plt.show()