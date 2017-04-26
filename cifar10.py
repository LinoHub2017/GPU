# plot ad hoc CIFAR10

from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
"""
#load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# create a grid of 3x3 images

for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))

# show the plot
pyplot.show()
"""
import h5py
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation, Dense
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import matplotlib.pyplot as plt


import numpy as np
import scipy.misc

seed = 7
numpy.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# create a model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile model
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.8, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=100, shuffle=True)

#history = model.fit(X_train, y_train, validation_split=0.25, epochs=epochs, batch_size=50, shuffle=True)

# list all history
print(history.history.keys())

# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

    
print(history.history['acc'])
print(history.history['val_acc'])

 
def load_and_scale_imgs():
   #img_names = ['ship.jpg', 'truck.jpg', 'cat.jpg', 'car.jpeg', 'plane.jpg', 'horse.jpg', 'dog.jpg']
   img_names = ['ship.jpg','plane.jpg']
   imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                        (2, 0, 1)).astype('float32')
           for img_name in img_names]
   return np.array(imgs) / 255


if __name__ == '__main__':
 
    imgs = load_and_scale_imgs()
    predictions = model.predict_classes(imgs)
    print(predictions)
    

    # summarize history for accuracy
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




























