import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# DIR = 'C:\\Users\\aycae\\PycharmProjects\\AI\\best-artworks-of-all-time'
# CATEGORIES = ["Edgar_Degas","Pablo_Picasso","Vincent_van_Gogh"]
# training_data=[]
# IMG_SIZE = 300
# for category in CATEGORIES:
#      path = os.path.join(DIR, category)
#
#      for img in os.listdir(path):
#          class_no = CATEGORIES.index(category)
#          try:
#              img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
#              new_array = cv2.resize(img_array, (IMG_SIZE, (IMG_SIZE)))
#              training_data.append([new_array, class_no])
#          except Exception as e:
#              print("error")
#
# random.shuffle(training_data)
#
# X=[]
# y=[]
#
# for painting,label in training_data:
#     X.append(painting)
#     y.append(label)
#
# X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# y = np.array(y)
#
# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X=X/255.0
model=tf.keras.models.Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X,y,batch_size=32,validation_split=0.3,epochs=5)








