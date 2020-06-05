import keras
import tensorflow as tf
import keras.layers.normalization
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle
import random
import sklearn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DIR = 'C:\\Users\\aycae\\PycharmProjects\\AI\\best-artworks-of-all-time'
CATEGORIES = ["Edgar_Degas","Pablo_Picasso","Vincent_van_Gogh"]
training_data=[]
IMG_SIZE = 300
for category in CATEGORIES:
     path = os.path.join(DIR, category)

     for img in os.listdir(path):
         class_no = CATEGORIES.index(category)
         try:
             img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
             new_array = cv2.resize(img_array, (IMG_SIZE, (IMG_SIZE)))
             training_data.append([new_array, class_no])
         except Exception as e:
             print("error")

random.shuffle(training_data)

X=[]
y=[]

for painting,label in training_data:
    X.append(painting)
    y.append(label)

X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle= True)


X=X/255.0

model=tf.keras.models.Sequential()

model.add(Conv2D(64, (3, 3),input_shape=X.shape[1:],activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),input_shape=X.shape[1:],activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3),activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(16,activation=tf.nn.relu))
model.add(Flatten())
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

aug = ImageDataGenerator()
BS=32
model.fit(aug.flow(X_test, y_test, batch_size = BS),validation_data = (X_train, y_train), steps_per_epoch = 20,epochs =10)

model.save("painting_classifier.model")

the_model=tf.keras.models.load_model("painting_classifier.model")

predict=the_model.predict([X_test])

print("The prediction is:")
p=np.argmax(predict[15])
if p == 0:
    print("Edgar Degas")
elif p == 1:
    print("Pablo Picasso")
elif p == 2:
    print("Vincent Van Gogh")
else:
    print("what")

plt.imshow(cv2.cvtColor(X_test[15], cv2.COLOR_BGR2RGB))
plt.show()

