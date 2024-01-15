#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


# In[ ]:


file_dir_1 = '/content/drive/MyDrive/atr/train-00000-of-00002-9e861cba931f46ba.parquet'
file_dir_2 = '/content/drive/MyDrive/atr/train-00001-of-00002-118ddee16eed69a9.parquet'


# In[ ]:


data_1 = pd.read_parquet(file_dir_1)
data_2 = pd.read_parquet(file_dir_2)


# In[ ]:


all_data = pd.concat([data_1, data_2], axis=0, ignore_index=True)
print(data_1.shape)
print(data_2.shape)
print(all_data.shape)


# In[ ]:


data_1[:2500].shape


# In[ ]:


import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from io import BytesIO  # Add this import

def prepare_data(df, input_size=(256, 256)):
    # Step 1: Decode byte strings
    df['image'] = df['image'].apply(lambda x: np.array(Image.open(BytesIO(x['bytes'])).convert("RGB")))
    df['mask'] = df['mask'].apply(lambda x: np.array(Image.open(BytesIO(x['bytes'])).convert("L")))

    # Step 2: Resize images and masks
    df['image'] = df['image'].apply(lambda x: np.array(Image.fromarray(x).resize(input_size)))
    df['mask'] = df['mask'].apply(lambda x: np.array(Image.fromarray(x).resize(input_size, resample=Image.NEAREST)))

    # Step 3: Split into X and y
    X = np.stack(df['image'].values)
    y = np.stack(df['mask'].values)

    # Step 4: Normalize pixel values
    X = X / 255.0

    # Step 5: One-hot encode target masks
    y = to_categorical(y, num_classes=18)  # Replace NUM_CLASSES with the number of classes in your segmentation task

    return X, y

# Example usage:
X, y = prepare_data(data_1[:2500])


# In[ ]:


x_train, x_test = X[0:int(X.shape[0]*0.9)], X[int(X.shape[0]*0.9)::]
y_train, y_test = y[0:int(X.shape[0]*0.9)], y[int(X.shape[0]*0.9)::]


# In[ ]:


import gc

del X, y
gc.collect()  # Явный вызов сборщика мусора


# In[ ]:


x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)


# In[ ]:


from keras import layers
import keras

def get_model_v2(img_size, num_classes):

  inputs = keras.Input(shape=img_size + (3,))

  conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = layers.Dropout(0.5)(conv4)
  pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = layers.Dropout(0.5)(conv5)

  up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(drop5))
  merge6 = layers.concatenate([drop4,up6], axis = 3)
  conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv6))
  merge7 = layers.concatenate([conv3,up7], axis = 3)
  conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv7))
  merge8 = layers.concatenate([conv2,up8], axis = 3)
  conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv8))
  merge9 = layers.concatenate([conv1,up9], axis = 3)
  conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

  outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)
  model = keras.Model(inputs, outputs)

  return model



# In[ ]:


get_ipython().system('pip install segmentation_models')


# In[ ]:


#!pip install segmentation_models
get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import segmentation_models as sm
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


# In[ ]:


# Build model
model2 = get_model_v2((256, 256), 18)
model3 = get_model_v2((256, 256), 18)
#model.summary()


model2.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=[focal_loss],
    metrics = ["accuracy",sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
)

model3.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=[total_loss],
    metrics = ["accuracy",sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
)


# In[ ]:


history = model2.fit(x_train_tensor, y_train_tensor, batch_size=2, epochs=40)


# In[ ]:


history = model3.fit(x_train_tensor, y_train_tensor, batch_size=2, epochs=40)


# In[ ]:


model2.save('my_model_focal_loss.h5')
model3.save('my_model_total_loss.h5')


# In[ ]:


pred2 = model2.predict(x_test_tensor[:5])
pred3 = model3.predict(x_test_tensor[:5])

pred2_img_arr_0 = np.argmax(pred2[0], axis=2).reshape(256, 256, 1)
pred3_img_arr_0 = np.argmax(pred3[0], axis=2).reshape(256, 256, 1)

import matplotlib.pyplot as plt
import numpy as np


TempLake = pred2_img_arr_0
im = plt.imshow(TempLake, cmap='hot', interpolation='none')
plt.colorbar(im)
plt.show()

TempLake = pred3_img_arr_0
im = plt.imshow(TempLake, cmap='hot', interpolation='none')
plt.colorbar(im)
plt.show()

pred2_img_arr_0 = np.argmax(y_test_tensor[0], axis=2).reshape(256, 256, 1)
TempLake = pred2_img_arr_0
im = plt.imshow(TempLake, cmap='hot', interpolation='none')
plt.colorbar(im)
plt.show()

print('---------focal-loss----------')
print(focal_loss(y_test_tensor[0], pred2[0]))
print(focal_loss(y_test_tensor[0], pred3[0]))
print('---------diss-loss----------')
print(dice_loss(y_test_tensor[0], pred2[0]))
print(dice_loss(y_test_tensor[0], pred3[0]))
print('---------total-loss----------')
print(total_loss(y_test_tensor[0], pred2[0]))
print(total_loss(y_test_tensor[0], pred3[0]))



# In[ ]:


model2.save('/content/drive/MyDrive/ATR_SEG_CLOTHES/my_model_focal_loss.keras')
model3.save('/content/drive/MyDrive/ATR_SEG_CLOTHES/my_model_total_loss.keras')


# In[ ]:




