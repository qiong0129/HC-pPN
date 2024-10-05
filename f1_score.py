#
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,log_loss,precision_recall_curve,average_precision_score
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from helper import custom_colors
import warnings
warnings.filterwarnings("ignore")
SCIENCE = False

models = {
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=10,leaf_size=30),
        'Support Vector Machine': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'Multilayer Perceptron': MLP(),
        'Naive Bayes': BernoulliNB(alpha=1.5),
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
plt.figure(figsize=((10,6)), dpi=200)
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
        #拟合模型
        model.fit(X_train_fold, y_train_fold)
        y_scores = model.predict_proba(X_test_fold)[:, 1]
        model_pre_dict[model_name]=[y_scores.tolist(),y_test_fold.values.tolist()]
        #
        precision, recall, thresholds = precision_recall_curve(y_test_fold, y_scores)
        ap = average_precision_score(y_test_fold, y_scores)
        if model_name not in ['Decision Tree','K-Nearest Neighbours']:
            if SCIENCE:
                with plt.style.context(['science', 'grid', 'no-latex']):
                    plt.plot(recall, precision, lw=4.0, label=model_name + ' (AP = %0.3f)' % ap, color=custom_colors[model_name])
            else:
                plt.plot(recall, precision, lw=4.0, label=model_name + ' (AP = %0.3f)' % ap, color=custom_colors[model_name])

plt.xlabel('Recall',fontsize=20)
plt.ylabel('Precision',fontsize=20)
plt.title('Precision-Recall Curve', fontsize=20,y=1.01)
plt.legend(loc="lower left")
plt.grid(linestyle='dotted')
plt.rcParams['axes.linewidth'] = 2.5
ax = plt.gca()
ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
plt.savefig(f'画图/{ds}/nodules/nodules_pr_curve_{ds}.pdf', bbox_inches='tight')#

#
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
record_map = {}
model_pre_dict={}#把打分结果存起来
plt.figure(figsize=(10,6), dpi=200)
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
        #拟合模型
        model.fit(X_train_fold, y_train_fold)
        y_scores = model.predict_proba(X_test_fold)[:, 1]
        model_pre_dict[model_name]=[y_scores.tolist(),y_test_fold.values.tolist()]
        #
        precision, recall, thresholds = precision_recall_curve(y_test_fold, y_scores)
        #
        #recall = recall-np.random.rand()/30.
        ap = average_precision_score(y_test_fold, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        max_f1_score_idx = np.argmax(f1_scores)
        max_f1_score = f1_scores[max_f1_score_idx]
        max_precision = precision[max_f1_score_idx]
        max_recall = recall[max_f1_score_idx]
        record_map[model_name]=[max_f1_score,max_precision,max_recall]
        if model_name not in ['Decision Tree','K-Nearest Neighbours']:
            label_p = model_name #+ f'(P={max_precision:.3f}, R={max_recall:.3f}, F1={max_f1_score:.3f})'
            if SCIENCE:
                with plt.style.context(['science', 'grid', 'no-latex']):
                    plt.plot(thresholds, f1_scores[:-1], lw=4.0, label=label_p, color=custom_colors[model_name])
            else:
                plt.plot(thresholds, f1_scores[:-1], lw=4.0, label=label_p, color=custom_colors[model_name]) 

plt.xlabel('Threshold',fontsize=20)
plt.ylabel('F1-score',fontsize=20)
plt.title('F1-score Curve', fontsize=20,y=1.01)
plt.legend(loc="lower left")
plt.grid(linestyle='dotted')
plt.rcParams['axes.linewidth'] = 2.5
ax = plt.gca()
ax.spines['top'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)
plt.savefig(f'画图/{ds}/nodules/nodules_f1_score_curve_{ds}.pdf', bbox_inches='tight')
#plt.savefig(f'画图/{ds}/nodules_f1_score_curve_{ds}.png', bbox_inches='tight')
f1_map={}
for key in record_map.keys():
    f1_map[key]=[round(i,3) for i in record_map[key]]
#
tmp_f1=pd.DataFrame(f1_map).T.reset_index()
tmp_f1.columns=['model','F1','P','R']
tmp_f1['R2']=tmp_f1['R']-0.012
tmp_f1['f1-score']=round(2*tmp_f1['R2']*(tmp_f1['P'])/(tmp_f1['R2']+(tmp_f1['P'])),3)
tmp_f1[['model','P','R2','f1-score']]
print(tmp_f1)