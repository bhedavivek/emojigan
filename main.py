import cv2
import json
import numpy as np
import os
from keras import backend as K
from keras.initializers import RandomNormal
from keras.optimizers import RMSprop
from keras.layers import Input, Conv2D, LeakyReLU, Dropout, Flatten, MaxPool2D, Dense, Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model

# Loading JSON Config
print "----- Loading Config -----"
data = json.load(open("processed.json", "r"))


# Loading Images
print "----- Loading Images -----"
images = None
for emoji in data:
	img = cv2.imread(emoji["location"])
	if images is None:
		images = np.array([img])
	else:
		images = np.vstack((images, [img]))

def wasserstein(y_true, y_pred):
	return K.mean(y_true * y_pred)

def make_discriminator():
	# Weight Initializer
	weight_init = RandomNormal(mean=0.0, stddev=0.2)

	img = Input(shape=(64, 64, 3), name="input_image")

	x = Conv2D(16, (3,3), padding="same", name="conv_1", kernel_initializer=weight_init)(img)
	x = LeakyReLU()(x)
	x = MaxPool2D(pool_size=2)(x)
	x = Dropout(0.25)(x)

	x = Conv2D(32, (3,3), padding="same", name="conv_2", kernel_initializer=weight_init)(x)
	x = LeakyReLU()(x)
	x = MaxPool2D(pool_size=2)(x)
	x = Dropout(0.25)(x)

	x = Conv2D(64, (3,3), padding="same", name="conv_3", kernel_initializer=weight_init)(x)
	x = LeakyReLU()(x)
	x = MaxPool2D(pool_size=2)(x)
	x = Dropout(0.25)(x)

	x = Flatten()(x)

	fake_prob = Dense(1, activation="sigmoid", name="fake_prob")(x)

	return Model(inputs=[img], outputs=[fake_prob], name="Discriminator")

def make_generator():
	# Weight Initializer
	weight_init = RandomNormal(mean=0.0, stddev=0.2)

	# Noise Input
	z = Input(shape=(100,), name="input_noise")

	# Dense Layer
	x = Dense(8*8*64)(z)
	x = LeakyReLU()(x)
	x = Reshape((8,8,64))(x)

	# Deconvolution Layer
	x = UpSampling2D(size=(2,2))(x)
	x = Conv2DTranspose(32, (3,3), padding="same", kernel_initializer=weight_init)(x)
	x = LeakyReLU()(x)

	# Deconvolution Layer
	x = UpSampling2D(size=(2,2))(x)
	x = Conv2DTranspose(16, (3,3), padding="same", kernel_initializer=weight_init)(x)
	x = LeakyReLU()(x)

	# Deconvolution Layer
	x = UpSampling2D(size=(2,2))(x)
	fake_image = Conv2DTranspose(3, (3,3), padding="same", kernel_initializer=weight_init, activation='tanh')(x)

	return Model(inputs=[z], outputs=[fake_image], name="Generator")


discriminator = make_discriminator()
print discriminator.summary()
discriminator.compile(optimizer=RMSprop(lr=0.00005), loss=wasserstein)

generator = make_generator()
print generator.summary()