from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import scienceplots
import os
import gc
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

models = {
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbours': KNeighborsClassifier(),
        'Support Vector Machine': SVC(),
        'Naïve Bayes': BernoulliNB(),
        'Decision Tree': DecisionTreeClassifier(min_samples_split=5),
        'Random Forest': RandomForestClassifier(n_estimators=100,min_samples_split=5),
        'LightGBM':LGBMClassifier(max_depth=5,
                                  learning_rate=0.1,
                                  n_estimators=500,
                                  num_leaves=2 ** 5-1
                                 ),
        'GBDT':GradientBoostingClassifier(max_depth=5,
                                learning_rate=0.05,
                                n_estimators=550)
}

df_279 = pd.read_csv('data_processed/df_279.csv')
train_df = pd.read_csv('data_processed/train_df.csv')
df_saliva = pd.read_csv('data_processed/df_saliva.csv')
#
g_frts=[]
for k in df_279.columns:
    if k in df_saliva.columns and k.startswith('g__'):
        g_frts.append(k)
len(g_frts)
select_frts=g_frts[::]
print('len(select_frts):', len(select_frts))
train_x = train_df[select_frts].copy()
train_y = train_df["PN"]
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_x)
print(train_x.shape,train_y.shape)
# train_x = train_x.values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024) 
results = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, train_y, cv=kf, scoring='roc_auc')
    results[model_name] = scores.mean()
    #break
#
df_auc = pd.DataFrame.from_dict(results, orient='index', columns=['auc_score'])
df_auc.reset_index(inplace=True)
df_auc.columns = ['model type', 'auc_score']
df_auc.sort_values(['auc_score'])