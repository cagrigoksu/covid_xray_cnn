# Covid 19 Chest X-Ray Prediction Using CNN over Imbalanced Dataset
# Cagri Goksu USTUNDAG 

from pathlib import Path
import pandas as pd

data_dir = Path("dataset/")
train_dir = data_dir/'train'
val_dir = data_dir/'val'

#* loads data, returns paths and labels dataframe
def load_data(directories):  

    dfs = {}
    for dir in directories:

        path_and_label = {}
        normal_dir = dir/'Normal'
        pneumonia_dir = dir/'Pneumonia'
        covid_dir = dir/'Covid' 

        normal_paths = normal_dir.glob('*.jpeg')
        pneumonia_paths = pneumonia_dir.glob('*.jpeg')
        covid_paths = covid_dir.glob('*.jpeg') 

        for path in normal_paths:
            path_and_label[path] = 'Normal'
        for path in pneumonia_paths:
            path_and_label[path] = 'Pneumonia'
        for path in covid_paths:
            path_and_label[path] = 'Covid'

        path_and_label_df = pd.DataFrame(path_and_label.items())
        path_and_label_df = path_and_label_df.rename(columns = { 0: 'path', 1: 'label'})

        #* shuffle dataset and reset index
        path_and_label_df = path_and_label_df.sample(frac = 1).reset_index(drop = True)

        dfs[dir] = path_and_label_df

    return dfs


dirs = [ train_dir, val_dir]
data_dfs = load_data(dirs)

train_df = data_dfs[train_dir]
val_df = data_dfs[val_dir]

plt.subplot(1, 2, 1)
plt.bar(train_df['label'].value_counts().index,train_df['label'].value_counts().values, color = 'r', alpha = 0.7)
plt.xlabel("Case Types")
plt.ylabel("Number of Cases")
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.bar(val_df['label'].value_counts().index,val_df['label'].value_counts().values, color = 'g', alpha = 0.7)

plt.xlabel("Case Types")
plt.ylabel("Number of Cases")
plt.grid(axis='y')
