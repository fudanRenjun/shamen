import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

# 导入CSV文件
data = pd.read_csv('E:/RS/sm/ALL-Feature/MLP/MLP-20.csv')

# 提取标签和特征
y = data.iloc[:, 0]  # 第一列为标签
X = data.iloc[:, 1:]  # 其他列为特征

# 准备存储AUC值
auc_results = {}

# 按顺序累加特征并训练模型
for i in range(1, 21):
    X_subset = X.iloc[:, :i]  # 选择前i个特征
    auc_per_class = []

    for label in range(8):  # 假设标签范围为0-7
        # 创建二分类标签
        y_binary = (y == label).astype(int)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_binary, test_size=0.3, random_state=42)

        # 训练XGBoost模型
        model = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', max_iter=500)
        model.fit(X_train, y_train)

        # 预测概率
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算AUC
        auc = roc_auc_score(y_test, y_pred_prob)
        auc_per_class.append(auc)

    # 存储每个特征数量对应的AUC值和平均AUC值
    avg_auc = np.mean(auc_per_class)
    auc_results[i] = {
        'AUC': auc_per_class,
        '平均AUC': avg_auc
    }

# 将结果转换为DataFrame并输出为表格
auc_df = pd.DataFrame(auc_results).T

# 检查实际列数
print(auc_df)

# 根据实际列数设置列名
if auc_df.shape[1] == 8:  # 只有AUC值
    auc_df.columns = [f'标签{label}' for label in range(8)]
elif auc_df.shape[1] == 9:  # 包含平均AUC
    auc_df.columns = [f'标签{label}' for label in range(8)] + ['平均AUC']

auc_df.reset_index(inplace=True)
auc_df.rename(columns={'index': '特征数量'}, inplace=True)
print(auc_df)
auc_df.to_csv('E:/RS/sm/ALL-Feature/MLP/MLP-20-ROC.csv', index=False, encoding='utf-8-sig')