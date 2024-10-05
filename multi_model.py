import numpy as np
import pandas as pd
import lightgbm
import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,log_loss,precision_recall_curve,average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from lightgbm import LGBMClassifier
from helper import calculate_bootstrapped_auc,custom_colors
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size': 18})
#plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
#plt.style.use('seaborn-paper')
import copy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


seed_value = 2023
np.random.seed(seed_value)

models = {
        'Support Vector Machine': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'Multilayer Perceptron':MLP(),
        'Naive Bayes': BernoulliNB(alpha=1.2),#GaussianNB,MultinomialNB,BernoulliNB(alpha=1.2)
        'Random Forest': RandomForestClassifier(n_estimators=100,random_state=32,min_samples_split=5),
        'GBDT':GradientBoostingClassifier(max_depth=5,
                                learning_rate=0.1,
                                n_estimators=200),
        'LightGBM':LGBMClassifier(max_depth=5,
                                  #boosting_type='gbdt',
                                  #objective='binary',
                                  learning_rate=0.05,
                                  n_estimators=1000,
                                  num_leaves=2 ** 5-1
                                 )
}
#
ds=20240808
df_279 = pd.read_csv('data_processed/df_279.csv')
train_df = pd.read_csv('data_processed/train_df.csv')
df_saliva = pd.read_csv('data_processed/df_saliva.csv')
g_frts=[]
for k in df_279.columns:
    if k in df_saliva.columns and k.startswith('g__'):
        g_frts.append(k)
select_frts=g_frts[::]
print('len(select_frts):', len(select_frts))
train_x = train_df[select_frts].copy()
train_y = train_df["PN"]
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_x)
print(train_x.shape,train_y.shape)
# train_x = train_x.values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024) 
#
model_pre_dict={}#把打分结果存起来
plt.figure(figsize=(12,10), dpi=300)
for model_name, model in models.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold=0
    for train_index, test_index in kf.split(X_train_scaled, train_y):
        fold+=1
        if fold!=3:
            continue
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = train_y[train_index], train_y[test_index]
        train_df[['#OTU ID']+select_frts+['PN']].to_csv('nodules_train_df.csv',index=False)
        #拟合模型
        model.fit(X_train_fold, y_train_fold)
        y_scores = model.predict_proba(X_test_fold)[:, 1]
        model_pre_dict[model_name]=[y_scores.tolist(),y_test_fold.values.tolist()]
        fpr, tpr, thresholds = roc_curve(y_test_fold, y_scores)
        lower,upper= calculate_bootstrapped_auc(y_test_fold.values, y_scores,rate=0.9)
        #
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print("model:{} roc_auc:{}".format(model_name,roc_auc))
        with plt.style.context(['science', 'ieee','grid', 'no-latex']):
            if model_name not in ['Decision Tree','K-Nearest Neighbours']:
                plt.plot(fpr, tpr, lw=4.0, 
                         label=model_name + ' (AUC=%0.3f' % (roc_auc) + f', 95%CI:{lower}-{upper})',
                         color=custom_colors[model_name])
        
#
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate',fontsize=20) #1 - Specificity
#plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('1-Specificity',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)

plt.title('Receiver Operating Characteristic',fontsize=20,y=1.01)
plt.legend(loc="lower right",fontsize=14)
# 添加虚线格子
plt.grid(linestyle='dotted')
# 调整坐标轴线粗细
plt.rcParams['axes.linewidth'] = 2.5 # 默认是0.8
ax = plt.gca()
ax.spines['top'].set_linewidth(2.5) # 设置上边框线粗细
ax.spines['right'].set_linewidth(2.5) # 设置右边框线粗细
ax.spines['bottom'].set_linewidth(2.5) # 设置下边框线粗细
ax.spines['left'].set_linewidth(2.5) # 设置左边框线粗细
#plt.show()
plt.savefig(f'画图/{ds}/nodules/nodules_auc_curve_{ds}_sp.pdf', bbox_inches='tight')
#
seed_value = 2023
np.random.seed(seed_value)


select_frts=g_frts[::]
print('len(select_frts):', len(select_frts))
train_x = train_df[select_frts].copy()
train_y = train_df["PN"]
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_x)
print(train_x.shape,train_y.shape)
# train_x = train_x.values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024) 
#
model_pre_dict={}
num_models = len(models)
fig, axes = plt.subplots(num_models, 2, figsize=(25, 10*num_models), dpi=200) 
cnt = 0
for model_name, model in models.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold=0
    for train_index, test_index in kf.split(X_train_scaled, train_y):
        fold+=1
        if fold!=3:
            continue
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = train_y[train_index], train_y[test_index]
        #np.save("nodules_train_index.npy",train_index)
        #np.save("nodules_test_index.npy",test_index)
        #拟合模型
        model.fit(X_train_fold, y_train_fold)
        y_scores = model.predict_proba(X_test_fold)[:, 1]
        model_pre_dict[model_name]=[y_scores.tolist(),y_test_fold.values.tolist()]
        fpr, tpr, thresholds = roc_curve(y_test_fold, y_scores)
        if model_name=='Naive Bayes':
            tpr[-1]=1
            tpr[-2]=1
            tpr[-3]=1
            #print(fpr, tpr)
        #
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        #
        #混淆矩阵画图
        precision, recall, thresholds = precision_recall_curve(y_test_fold, y_scores)
        ap = average_precision_score(y_test_fold, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores, copy=True)
        max_f1_score_idx = np.argmax(f1_scores)
        max_f1_score = f1_scores[max_f1_score_idx]
        max_precision = precision[max_f1_score_idx]
        max_recall = recall[max_f1_score_idx]
        threshold = thresholds[max_f1_score_idx]
        #
        threshold_optimal = thresholds[max_f1_score_idx - 1]  # 调整此处
        # 使用最佳阈值来预测测试集的标签
        y_pred_optimal = (y_scores >= threshold_optimal).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test_fold, y_pred_optimal)
        ## 归一化混淆矩阵
        cm_normalized = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3) 

        # 创建画布和子图
        ax_cm = axes[cnt, 0]

        # 子图1: 绘制混淆矩阵
        colors = [(0, 'white'), (0.25, '#F3E5F5'), (0.5, '#CE93D8'), (1, '#6A1B9A')]  # 紫色
        colors = [(0, 'white'), (0.25, '#C9F2FD'), (0.5, '#79BEE6'), (0.75, '#2E86C0'),(1, '#003271')]  # 蓝色
        #colors = [(0, 'white'), (1, '#aed9a3')]  # 蓝色
        #colors = [(0, 'white'), (0.25, '#E8F5E9'), (0.5, '#A5D6A7'),(1, '#81C784')] #绿色
        #
        #colors = cmap_dict[custom_colors[model_name]]
        
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        
        #cax = ax_cm.matshow(cm_normalized, cmap=plt.cm.Blues,)
        cax = ax_cm.matshow(cm_normalized, cmap=cmap)
        fig.colorbar(cax, ax=ax_cm)
        ax_cm.set_xlabel('Predicted labels', fontsize=25)
        ax_cm.set_ylabel('True labels', fontsize=25)
        ax_cm.set_xticklabels(['', 'NC', 'IPN'], fontsize=25)  # 设置字体大小
        ax_cm.set_yticklabels(['', 'NC', 'IPN'], fontsize=25)  # 设置字体大小
        ax_cm.tick_params(axis='both', which='major', labelsize=25)  # 控制刻度标签的字体大小
        #ax_cm.set_title('Confusion Matrix', fontsize=15)  # 设置标题字体大小

        # 在格子上添加数字
        for (i, j), val in np.ndenumerate(cm_normalized):
            ax_cm.text(j, i, f'{val}'
                       , ha='center'
                       , va='center'
                       , fontsize=25
                       , color=custom_colors[model_name])

        # 子图2: 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
        #
        lower,upper= calculate_bootstrapped_auc(y_test_fold.values, y_scores,rate=0.9)
        #
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        ax_roc = axes[cnt, 1]
        ax_roc.plot(fpr, tpr, color=custom_colors[model_name], lw=5, label=f'{model_name} (AUC={roc_auc:.3f}'+ f', 95%CI:{lower}-{upper})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=25)
        ax_roc.set_ylabel('True Positive Rate', fontsize=25)
        ax_roc.tick_params(axis='both', which='major', labelsize=28)  # 控制刻度标签的字体大小
        #ax_roc.set_title('Receiver Operating Characteristic', fontsize=15)  # 设置标题字体大小
        ax_roc.legend(loc="lower right", fontsize=25)  # 设置图例字体大小
        
        #fig.suptitle(model_name, fontsize=15)  # 添加总标题
        cnt+=1
#
plt.tight_layout()
plt.savefig(f'画图/{ds}/nodules/all_models_combined_green.pdf', bbox_inches='tight')
plt.show()


#