from keras.models import Sequential
from keras.layers import Reshape, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Flatten, Dense
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.initializers import TruncatedNormal
from keras import backend as K
import numpy as np
from PIL import Image
import argparse
import math
import json
import cv2
import scipy.misc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def generator_model():
    model = Sequential()
    model.add(Dense(512*4*4, input_shape=(10,), kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((4,4,512)))

    model.add(Conv2DTranspose(256, (7,7), strides=(2,2), padding="same", kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(128, (7,7), strides=(2,2), padding="same", kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, (7,7), strides=(2,2), padding="same", kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(3, (7,7), strides=(2,2), padding="same", kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(Activation('tanh'))
    
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (7, 7),strides=(2,2),padding='same',input_shape=(64, 64, 3), kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation(LeakyReLU(0.2)))

    model.add(Conv2D(128, (7, 7), padding="same", strides=(2,2), kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation(LeakyReLU(0.2)))

    model.add(Conv2D(256, (7, 7), padding="same", strides=(2,2), kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation(LeakyReLU(0.2)))

    model.add(Conv2D(512, (7, 7), padding="same", strides=(2,2), kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation(LeakyReLU(0.2)))

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=TruncatedNormal(mean=0, stddev=0.02), bias_initializer='zeros'))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, :]
    return image

def load_images():
    # Loading JSON Config
    print "----- Loading Config -----"
    data = json.load(open("processed.json", "r"))


    # Loading Images
    print "----- Loading Images -----"
    images = None
    for emoji in data:
        img = scipy.misc.imread(emoji["location"]).astype(np.float32)
        # img = cv2.normalize(img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img / 127.5 - 127.5
        if images is None:
            images = np.array([img])
        else:
            images = np.vstack((images, [img]))

    return images


def train(images):
    BATCH_SIZE = 16
    NOISE_SIZE = 10
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g_optim = Adam(lr=0.00002, beta_1=0.25)
    d.trainable=True
    d.compile(loss="binary_crossentropy", optimizer=g_optim)
    g = generator_model()
    d.trainable=False
    d_on_g = generator_containing_discriminator(g, d)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)
    
    for epoch in range(1000000):
        
        list_disc_real_loss = []
        list_disc_fake_loss = []
        # Train Discriminator
        for index in range(int(82/BATCH_SIZE)):
            d.trainable = True
            for l in d.layers:
                l.trainable = True
            noise = np.random.uniform(-1, 1,  size=(BATCH_SIZE,NOISE_SIZE))
            p = np.random.permutation(BATCH_SIZE)
            image_batch = images[p]
            noise_batch = g.predict(noise)
            disc_real_loss = d.train_on_batch(image_batch, np.ones(BATCH_SIZE))
            disc_fake_loss = d.train_on_batch(noise_batch, np.zeros(BATCH_SIZE))
            list_disc_real_loss.append(disc_real_loss)

            # Train Generator
            d.trainable = False
            for l in d.layers:
                l.trainable = False
            noise = np.random.uniform(-1, 1,  size=(BATCH_SIZE,NOISE_SIZE))
            gen_fake_loss = d_on_g.train_on_batch(noise, np.ones(BATCH_SIZE))

        # Print Progress
        print "Epoch {:d}".format(epoch)
        print "Disc_Real_Loss :", np.mean(list_disc_real_loss), ", Gen_Loss :", gen_fake_loss

        if epoch % 20 == 0:
            img = g.predict(np.random.uniform(low=-1, high=1, size=(1,10)))
            print img
            print img.shape
            location = "./samples/dcgan"
            scipy.misc.imsave(location+"/{}_sample.jpg".format(epoch),(img[0]+1.)*2.)

discriminator = discriminator_model()
print discriminator.summary()

generator = generator_model()
print generator.summary()

images = load_images()

train(images=images)