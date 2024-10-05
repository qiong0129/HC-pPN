import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from lightgbm import LGBMClassifier
from helper import calculate_bootstrapped_auc,custom_colors
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


seed_value = 2023
np.random.seed(seed_value)
auc_figsize=(10,6)
models = {
        
        'LightGBM_all':LGBMClassifier(max_depth=5,
                                  #boosting_type='gbdt',
                                  #objective='binary',
                                  learning_rate=0.05,
                                  n_estimators=1000,
                                  num_leaves=2 ** 5-1
                                 ),
        'LightGBM_top6':LGBMClassifier(max_depth=5,
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
cache_auc_x={}
model_pre_dict={}#把打分结果存起来
plt.figure(figsize=auc_figsize, dpi=300)
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
        #
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        lower,upper= calculate_bootstrapped_auc(y_test_fold.values, y_scores,rate=0.9)
        print("model:{} roc_auc:{}".format(model_name,roc_auc))
        plt.plot(fpr, tpr, lw=4.0, label='Top6 Microbial Species ' + ' (AUC=%0.3f' % (roc_auc)+ f', 95%CI:{lower}-{upper})',
                    color = 'tab:cyan',
                )
        
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
plt.savefig(f'画图/{ds}/nodules/nodules_auc_curve_{ds}_top6.pdf', bbox_inches='tight')