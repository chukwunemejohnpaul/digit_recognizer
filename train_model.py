__author__ = "Chukwuneme Tadinma Johnpaul"

import cv2
import keras
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


print("[INFO]-----Loading training and testing data")
# I load the MNIST dataset which i would be using to train the model
(x_train, y_train), (x_test,y_test) = keras.datasets.mnist.load_data()


# I create the model architecture with a keras functional Api. the model is 
# of a Lenet architecture with the addition of a batch normalization layer

print("[INFO]----Building model arhitecture")

inputs = keras.Input(shape=(1,28,28))
x = keras.layers.convolutional.Convolution2D(20,(3,3),padding="same")(inputs)
x = keras.layers.Activation("relu")(x)
x = keras.layers.pooling.MaxPooling2D((2,2),padding='same')(x)
x = keras.layers.convolutional.Convolution2D(50,(3,3),padding="same")(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.pooling.MaxPooling2D((2,2),padding='valid')(x)
x = keras.layers.BatchNormalization(axis=1)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(500,activation="relu")(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.Dense(50,activation="relu")(x)
outputs = keras.layers.Dense(10,activation="softmax")(x)

# The model is instantiated

model = keras.Model(inputs,outputs)
model.summary()

print("[INFO]----Compiling model")
model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])


print("[INFO]----preprocessing data")
x_train = np.reshape(x_train,(len(x_train),1,28,28))
x_test = np.reshape(x_test,(len(x_test),1,28,28))
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
print("[INFO]----done")

#datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
#dg = datagen.flow(x_train,y_train,batch_size=128)

print("[INFO]----Starting model training for 5 epochs \n")

history = model.fit(x_train,y_train,epochs=5,batch_size=128,shuffle=True,validation_data=(x_test,y_test))

print("[INFO]-----Model training complete")

print("[INFO]----Saving model")

model.save("mydigitrecognizer.h5")


print("[INFO]----showing model history")
print(history.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()