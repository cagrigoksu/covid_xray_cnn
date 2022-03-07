# Covid 19 Chest X-Ray Prediction Using CNN over Imbalanced Dataset
# Cagri Goksu USTUNDAG 


# download dataset from drive
"""

import zipfile
from google.colab import drive

#drive.mount('/content/drive/')

zip_ref = zipfile.ZipFile("/content/drive/MyDrive/datasetXL.zip", 'r')
zip_ref.extractall("/content")
zip_ref.close()

"""

# imports"""

import sys, time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import tensorflow as tf
from keras.preprocessing.image import img_to_array, save_img, ImageDataGenerator
import cv2
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
import gc
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import os

"""# load dataset"""

data_dir = Path("datasetXL/")
train_dir = data_dir/'train'
val_dir = data_dir/'val'

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
plt.ylabel("Number of Cases - Train")
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.bar(val_df['label'].value_counts().index,val_df['label'].value_counts().values, color = 'g', alpha = 0.7)

plt.xlabel("Case Types")
plt.ylabel("Number of Cases - Validation")
plt.grid(axis='y')

"""# prepare data, label"""

def prepare(df):
  data, labels = [], []

  for j, row in df.iterrows():

    img = cv2.imread(str(row['path']))  #* read image from path
    img = cv2.resize(img, (224,224))
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_.astype(np.float32)/255.

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

train_data, train_label = prepare(train_df)
val_data, val_label = prepare(val_df)

""" # flip and add to data"""

def generate_flipped(dir, label):

    input_img_list = dir.glob('*.jpeg')
    data = []
    lbl = []
    for img in input_img_list:
        sys.stdout.write('\r{0}'.format(str(img)))
        img_f = cv2.imread(str(img)) 
        img_f = cv2.resize(img_f, (224,224))
        img_ = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
        img_f = img_.astype(np.float32)/255. 
        flipped = tf.image.flip_left_right(img_f)
        data.append(flipped)
        
        if label == 'Covid':
            lbl.append(to_categorical(0, num_classes = 3))
        elif label == 'Normal': 
            lbl.append(to_categorical(1, num_classes = 3))
        else:
            lbl.append(to_categorical(2, num_classes = 3))

        sys.stdout.flush()

    data = np.array(data)
    lbl = np.array(lbl)
    
    return data, lbl

flipped_data, flipped_label = generate_flipped(train_dir/'Covid', 'Covid')

train_data_f = np.vstack([train_data, flipped_data])
train_label_f = np.vstack([train_label, flipped_label])

del train_data
del train_label
gc.collect()

#TODO add opposite degree to rotate
def generate_rotated_img(dir, label, angle, cropped = False): 

    input_img_list = dir.glob('*.jpeg')
    data = []
    lbl = []
    for img in input_img_list:
        sys.stdout.write('\r{0}'.format(str(img)))
        img_f = cv2.imread(str(img)) 
        img_f = cv2.resize(img_f, (224,224))
        img_ = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
        img_f = img_.astype(np.float32)/255. 

        if cropped:
            rotated = imutils.rotate(img_f, angle)
        else:
            rotated = imutils.rotate_bound(img_f, angle)
            
        rotated = cv2.resize(rotated, (224,224))
        data.append(rotated)
        
        if label == 'Covid':
            lbl.append(to_categorical(0, num_classes = 3))
        elif label == 'Normal': 
            lbl.append(to_categorical(1, num_classes = 3))
        else:
            lbl.append(to_categorical(2, num_classes = 3))

        sys.stdout.flush()

    data = np.array(data)
    lbl = np.array(lbl)
    
    return data, lbl

rotated_data, rotated_label = generate_rotated_img(train_dir/'Covid', 'Covid', 5)

train_data_r = np.vstack([train_data_f, rotated_data])
train_label_r = np.vstack([train_label_f, rotated_label])

del train_data_f
del train_label_f
gc.collect()


categories = np.argmax(train_label_r, axis=1)
cat_labels = []
for label in cat_labels:
    if label == 0:
        cat_labels.append('Covid')
    elif label == 1:
        cat_labels.append('Normal')
    else:
        cat_labels.append('Pneumonia')

label_df = pd.DataFrame(cat_labels)
plt.bar(label_df[0].value_counts().index, label_df[0].value_counts().values, color = 'r', alpha = 0.7)
plt.xlabel("Case Types")
plt.ylabel("Number of Cases")
plt.grid(axis='y')

"""# train model"""

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=64,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "content/vgg16_1.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)

hist = model.fit(train_data_f, train_label_f, 
                  validation_data= (val_data, val_label), 
                 epochs=5, callbacks=[checkpoint])

print(hist.history.keys())

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

# !mkdir -p saved # to create dir on colab
model.save('saved/model')
cnn_model = load_model("saved/model")

covid_img = cv2.imread("datasetXL/val/Covid/084.jpeg")  #* read image from path
covid_img = cv2.resize(covid_img, (224,224))
covid_img = covid_img.reshape(1, 224, 224, 3)
covid_img = np.asarray(covid_img)

normal_img = cv2.imread("datasetXL/val/Normal/NORMAL2-IM-1437-0001.jpeg") 
normal_img = cv2.resize(normal_img, (224,224))
normal_img = normal_img.reshape(1, 224, 224, 3)
normal_img = np.asarray(normal_img)

pneumonia_img = cv2.imread("datasetXL/val/Pneumonia/person1949_bacteria_4880.jpeg")  
pneumonia_img = cv2.resize(pneumonia_img, (224,224))
pneumonia_img = pneumonia_img.reshape(1, 224, 224, 3)
pneumonia_img = np.asarray(pneumonia_img)

covid_pred = cnn_model.predict(covid_img)
normal_pred = cnn_model.predict(normal_img)
pneumonia_pred = cnn_model.predict(pneumonia_img)

print(covid_pred, normal_pred, pneumonia_pred)