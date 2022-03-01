# Covid 19 Chest X-Ray Prediction Using CNN over Imbalanced Dataset
# Cagri Goksu USTUNDAG 

import sys, time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import tensorflow as tf
from keras.preprocessing.image import img_to_array, save_img, ImageDataGenerator
import cv2
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import keras
import numpy as np
from tensorflow.keras.utils import to_categorical

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

#! generates flipped of the images in the given path
#TODO do it by adding directly to the array instead of saving
def generate_flipped_image(dir):
    
  input_img_list = dir.glob('*.jpeg')

  for img in input_img_list:

    img_f = mpimg.imread(img) 
    img_f = img_to_array(img_f)    
    flipped = tf.image.flip_left_right(img_f)    
    save_img(str(img).split('.')[0]+'_flipped.jpeg', flipped)


generate_flipped_image(train_dir/'Covid')


#* Chart of the altered dataset
dir = [train_dir]
data_df = load_data(dir)

train_df_flipped = data_df[train_dir]

plt.bar(train_df_flipped['label'].value_counts().index,train_df_flipped['label'].value_counts().values, color = 'r', alpha = 0.7)
plt.xlabel("Case Types")
plt.ylabel("Number of Cases")
plt.grid(axis='y')

#* generate data using generator
#train_data = ImageDataGenerator().flow_from_directory(directory="datasetXL/train",target_size=(224,224))
#val_data = ImageDataGenerator().flow_from_directory(directory="datasetXL/val",target_size=(224,224))

#* defining a simple base conv and poolings
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=64,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

def prepare(df):
  data, labels = [], []

  for j, row in df.iterrows():

    img = cv2.imread(str(row['path']))  #* read image from path
    img = cv2.resize(img, (224,224))
    img = img.astype(np.float32)/255.

    sys.stdout.write('\r{0}'.format(str(j)))
    

    data.append(img)

    label_str = row['label']

    if label_str == 'Covid':
      labels.append(to_categorical(0, num_classes = 3))
    elif label_str == 'Normal': 
      labels.append(to_categorical(1, num_classes = 3))
    else:
      labels.append(to_categorical(2, num_classes = 3))

    sys.stdout.flush()

  return np.asarray(data), np.asarray(labels)

#* img, label pairs for train and test
t_data, t_label = prepare(train_df_flipped)
v_data, v_label = prepare(val_df)

batch_size = 32
hist = model.fit(t_data, t_label, steps_per_epoch= len(train_df_flipped)//batch_size, 
                  validation_data= (v_data, v_label), validation_steps= len(val_df)//batch_size, epochs=5)