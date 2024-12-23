import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sympy.physics.control.control_plots import matplotlib

# 显示中文字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 加载CSV文件
data = pd.read_csv('E:/RS/sm/XUN.csv')

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 2. 划分训练集和测试集
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=8)

# 3. 标准化数据
scaler = MinMaxScaler()
xtrain_s = scaler.fit_transform(xtrain)
xtest_s = scaler.transform(xtest)

# 使用随机森林进行建模以计算特征重要性
model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, max_depth=3, random_state=8)
model_rf.fit(xtrain_s, ytrain)

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
plt.savefig('E:/RS/sm/Top_20_Feature_Importances.png', format='png', bbox_inches='tight')
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
    model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, max_depth=3, random_state=8)
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

# 6. 保存结果到CSV文件
results_df = pd.DataFrame({
    'Number of Features': feature_counts,
    'ROC AUC': roc_aucs
})
results_df.to_csv('E:/RS/sm/ROC_AUC_vs_Features.csv', index=False)