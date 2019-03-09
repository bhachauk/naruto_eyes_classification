# import the necessary packages
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras import backend as K
from keras.utils.vis_utils import plot_model


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(len(class_)))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

class CNN1:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        cnn1 = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same', input_shape=inputShape))
        cnn1.add(MaxPooling2D(pool_size=(2, 2)))
        cnn1.add(Dropout(0.2))

        cnn1.add(Flatten())

        cnn1.add(Dense(128, activation='relu'))
        cnn1.add(Dense(len(class_), activation='softmax'))
        cnn1.add(Activation("softmax"))

        # return the constructed network architecture
        return cnn1

class CNN3:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        cnn3 = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        cnn3.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same', input_shape=inputShape))
        cnn3.add(MaxPooling2D((2, 2)))
        cnn3.add(Dropout(0.25))

        cnn3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        cnn3.add(MaxPooling2D(pool_size=(2, 2)))
        cnn3.add(Dropout(0.25))

        cnn3.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        cnn3.add(Dropout(0.4))

        cnn3.add(Flatten())

        cnn3.add(Dense(128, activation='relu'))
        cnn3.add(Dropout(0.3))
        cnn3.add(Dense(len(class_), activation='softmax'))

        cnn3.add(Activation("softmax"))

        # return the constructed network architecture
        return cnn3

class CNN4:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        cnn4 = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same', input_shape=inputShape))
        cnn4.add(BatchNormalization())

        cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(MaxPooling2D(pool_size=(2, 2)))
        cnn4.add(Dropout(0.25))

        cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(Dropout(0.25))

        cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(MaxPooling2D(pool_size=(2, 2)))
        cnn4.add(Dropout(0.25))

        cnn4.add(Flatten())

        cnn4.add(Dense(512, activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(Dropout(0.5))

        cnn4.add(Dense(128, activation='relu'))
        cnn4.add(BatchNormalization())
        cnn4.add(Dropout(0.5))

        cnn4.add(Dense(len(class_), activation='softmax'))
        cnn4.add(Activation("softmax"))

        # return the constructed network architecture
        return cnn4

class VGG_16:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        inputShape = (width, height, depth)

        # if we are using "channels first", update the input shape
        # K.set_image_dim_ordering('th')
        if K.image_data_format() == "channels_first":
            inputShape = (depth, width, height)

        print inputShape

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=inputShape))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(class_), activation='softmax'))
        return model

import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

EPOCHS = 100
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
MODULE_DIR = './modules/'
PLOT_DIR = './plots/'
class_ = {"sharingan": 0, "byakugan": 1, "sage": 2, "others": 3}
global_imgformat = (28, 28, 1)
gwidth, gheight, gdepth = global_imgformat

models = [
    ('CNN1', CNN1),
    ('LENET', LeNet),
    ('CNN3', CNN3),
    ('CNN4', CNN4),
    ('VGG16', VGG_16)
]

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath, 0)
    image = image[:, :, np.newaxis]
    image = cv2.resize(image, (gwidth, gheight))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = class_[label]
    labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print "[INFO] Loaded Labels Labels : {}".format(set(labels))
print "[INFO] Loaded Data Size : {}".format(len(data))

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.10, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=len(class_))
testY = to_categorical(testY, num_classes=len(class_))

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model

for name, model in models:
    print ("[INFO] compiling model... : {}".format(name))
    model = model.build(width=gwidth, height=gheight, depth=gdepth, classes=class_)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    plot_model(model, to_file="{}{}_schema.png".format(PLOT_DIR, name))

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=0)

    # save the model to disk
    modelfile = MODULE_DIR + name+'_naruto_eye.h5'
    print("[INFO] serializing network... and Model saved to the file Name : "), modelfile
    model.fit(data, to_categorical(labels, num_classes=len(class_)))
    model.save(modelfile)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy with the Net : {}".format(name))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    fileName = PLOT_DIR + name + '.png'
    #plt.savefig(fileName)