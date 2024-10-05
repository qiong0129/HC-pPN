import numpy as np
import pandas as pd
import os

if __name__=="__main__":
    #
    df_279 = pd.read_csv('data/df_279.csv')
    df_317 = pd.read_csv('data/df_317.csv')
    df_base = pd.read_csv('data/cancer_213_20231213.csv')
    df_base.rename(columns={'cancer': 'MPN'}, inplace=True)
    df_base=df_base.rename(columns={'性别': "gender",
                                '年龄': "age",
                                '个人肿瘤史':'personal_cancer_history',
                                '家族肿瘤史':'family_cancer_history',
                                '吸烟史':'smoke'})
    df_saliva=pd.read_excel('data/482_saliva.xlsx')
    df_saliva=df_saliva.transpose()#转置
    df_saliva.columns = df_saliva.iloc[0]
    df_saliva=df_saliva[1:].reset_index(drop=True)
    df_saliva.columns=['#OTU ID']+df_saliva.columns[1:].tolist()
    need_drop=[]
    for k in df_saliva.columns:
        if k in df_base.columns:
            if k!='#OTU ID':
                need_drop.append(k)
    df_base=df_base.drop(need_drop,axis=1)
    g_frts=[]
    for k in df_279.columns:
        if k in df_saliva.columns and k.startswith('g__'):
            g_frts.append(k)
    df=pd.merge(df_base,df_saliva,on='#OTU ID')
    # 找到数值列
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    # 找到object列
    object_cols = df.select_dtypes(include='object').columns
    #
    #除了 '#OTU ID','影像资料','危险分层' 这仨，其余的object列其实本身都是数字，所以可以直接转换成int类型
    for col in object_cols:
        if col not in ['#OTU ID','影像资料','危险分层']:
            df[col]=df[col].astype(int)
    object_cols = ['#OTU ID','影像资料','危险分层']
    #
    df[numeric_cols] = df[numeric_cols].fillna(-999)#数值为空的列，用-999来填充
    df['影像资料'] = df['影像资料'].fillna('')#文本用空字符串填充
    df_emmpy = pd.concat((df[df['危险分层'].isnull()],df[df['危险分层']=='不清']))
    df_clean = df[(df['危险分层'].notnull()) & (df['危险分层']!='不清')].reset_index(drop=True)
    df_clean['危险分层']=df_clean['危险分层'].astype(int)
    #
    train_df = df_clean
    train_df['PN']=(train_df['危险分层']>0).astype(int)
    train_df_nodules=train_df[train_df['PN']>0].reset_index(drop=True)
    #
    os.makedirs('data_processed/',exist_ok=True)
    train_df.to_csv('data_processed/train_df.csv', index=False)
    train_df_nodules.to_csv('data_processed/train_df_nodules.csv', index=False)
    df_279.to_csv('data_processed/df_279.csv', index=False)
    df_317.to_csv('data_processed/df_317.csv', index=False)
    df_saliva.to_csv('data_processed/df_saliva.csv', index=False)
