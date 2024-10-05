import numpy as np
import pandas as pd
import lightgbm
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,log_loss,precision_recall_curve,average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from lightgbm import LGBMClassifier
from helper import calculate_bootstrapped_auc,custom_colors
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import copy
import warnings
warnings.filterwarnings("ignore")


seed_value = 2023
np.random.seed(seed_value)

#
plt.rcParams['figure.dpi'] = 300
import shap
shap.initjs()  # notebook环境下，加载用于可视化的JS代码
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
X_train, X_test, y_train, y_test = train_test_split(train_x,
                                                    train_y,
                                                    test_size=0.2,
                                                    stratify=train_y,
                                                    random_state=2024)
#
model = LGBMClassifier(max_depth=6,
                      boosting_type='gbdt',
                      objective='binary',
                      learning_rate=0.1,
                      n_estimators=200,
                      num_leaves=2 ** 5-1
                     )
model.fit(X_train,y_train)
y_scores = model.predict_proba(X_test)[:,1]
#
auc = roc_auc_score(y_test, y_scores)
print(f"auc:{auc}")
#SHAP解释
explainer = shap.TreeExplainer(model)#model 为训练好的机器学习模型
shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
#

#matplotlib=True
# shap_plt = shap.force_plot(explainer.expected_value[0], shap_values[0][1],
#                            X_test.iloc[1],matplotlib=True,show = False)
# shap_plt.savefig(f'画图/{ds}/nodules_Xshap_plot_{sample_index}_{ds}.pdf', bbox_inches='tight')
# #from shap.plots import _force_matplotlib_html
# shap_plt = _force_matplotlib_html(explainer.expected_value[0], shap_values[0][sample_index],
#                                   X_test.iloc[sample_index], show=False)
sample_index = 10
shap.plots.force(explainer.expected_value[1], shap_values[1][sample_index]
                ,X_test.iloc[sample_index]
                ,matplotlib=True
                #,show=False
                ,figsize=(16,6)
                ,text_rotation=270)
#
plt.savefig(f'画图/{ds}/nodules_Xshap_plot_{sample_index}_{ds}.pdf', bbox_inches='tight')
#shap_plt.savefig(f'画图/{ds}/nodules/shap/shap_plot_{sample_index}_{ds}.svg', bbox_inches='tight')
#shap_plt.savefig(f'画图/{ds}/nodules/shap/shap_plot_{sample_index}_{ds}.png', bbox_inches='tight')

# 对多个样本进行综合解释并绘制图像
fig = plt.figure(figsize=(20,5),dpi=200)
#shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test,plot_type="dot")

fig.savefig(f'画图/{ds}/nodules/shap/nodules_shap_summary_plot_{ds}.pdf', bbox_inches='tight')