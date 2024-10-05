import os
#
save_dir = f'画图/{ds}'
os.makedirs(save_dir,exist_ok=True)
os.makedirs(save_dir+f'/nodules/',exist_ok=True)
os.makedirs(save_dir+f'/nodules/shap',exist_ok=True)