from sklearn.metrics import roc_curve
import sklearn
import random as np
#
cmap_dict = {
    'tab:blue': [(0, 'white'), (0.5, '#EDF2FB'), (1.0, '#85A7DC')],  # 蓝色渐变色
    'tab:orange': [(0, 'white'), (0.5, '#FFF3E0'), (1.0, '#FFCC80')],  # 橙色渐变色
    'tab:green': [(0, 'white'), (0.5, '#E8F5E9'), (1.0, '#A5D6A7')],  # 绿色渐变色
    'tab:red': [(0, 'white'), (0.5, '#FFCDD2'), (1.0, '#E57373')],  # 红色渐变色
    'tab:purple': [(0, 'white'), (0.5, '#F3E5F5'), (1.0, '#CE93D8')],  # 紫色渐变色
    'tab:brown': [(0, 'white'), (0.5, '#EFEBE9'), (1.0, '#D7CCC8')],  # 棕色渐变色
    'tab:pink': [(0, 'white'), (0.5, '#FCE4EC'), (1.0, '#F48FB1')],  # 粉色渐变色
    'tab:olive': [(0, 'white'), (0.5, '#F9FBE7'), (1.0, '#DCEDC8')],  # 橄榄色渐变色
    'tab:cyan': [(0, 'white'), (0.5, '#E0F7FA'), (1.0, '#80DEEA')]  # 青色渐变色
}
custom_colors = {
    'Mayo': 'tab:blue',
    'Mayo-BioEnhanced': 'tab:orange',
    'Decision Tree': 'tab:blue',
    'K-Nearest Neighbours': 'tab:orange',
    'Support Vector Machine': 'tab:green',
    'Logistic Regression': 'tab:red',
    'Multilayer Perceptron': 'tab:purple',
    'Naive Bayes': 'tab:brown',
    'Random Forest': 'tab:pink',
    'GBDT': 'tab:olive',
    'LightGBM': 'tab:cyan',
    'XGBoost': '#9467bd',  # 使用Hex颜色代码
    'AdaBoost': '#7f7f7f',  # 使用Hex颜色代码
    'CatBoost': '#bcbd22',  # 使用Hex颜色代码
    'Extra Trees': '#d62728',  # 使用Hex颜色代码
    'Gradient Boosting': '#9467bd',  # 使用Hex颜色代码
    'Lasso Regression': '#8c564b',  # 使用Hex颜色代码
    'Ridge Regression': '#2ca02c'  # 使用Hex颜色代码
}
lower_bound=5
upper_bound=95
def calculate_bootstrapped_auc(y_true, y_scores, rate):
    n_bootstraps = 500  # 设置bootstrapping的迭代次数
    auc_values = []
    for i in range(n_bootstraps):
        # 从预测结果和真实标签中随机选择样本
        indices = np.random.choice(np.arange(len(y_scores)), size=int(len(y_scores)*rate), replace=True)
        bootstrap_scores = y_scores[indices]
        bootstrap_true_labels = y_true[indices]

        # 计算ROC曲线的真阳性率和假阳性率
        bootstrap_fpr, bootstrap_tpr, _ = roc_curve(bootstrap_true_labels, bootstrap_scores)

        # 计算AUC值
        bootstrap_auc = sklearn.metrics.auc(bootstrap_fpr, bootstrap_tpr)
        auc_values.append(bootstrap_auc)

    # 计算AUC的置信区间
    lower = np.percentile(auc_values, lower_bound)
    upper = np.percentile(auc_values, upper_bound)
    return [round(lower,3),round(upper,3)]

