# Covid 19 Chest X-Ray Prediction Using CNN over Imbalanced Dataset
# Cagri Goksu USTUNDAG 

from pathlib import Path
import pandas as pd

data_dir = Path("datasetXL/")
train_dir = data_dir/'train'
val_dir = data_dir/'val'

#* loads data, returns paths and labels dataframe
def load_data(directories):

  path_and_label = {}

  for dir in directories:

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

  return path_and_label_df


dirs = [ train_dir, val_dir]
data = load_data(dirs)
data