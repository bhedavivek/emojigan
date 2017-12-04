import tensorflow as tf
import numpy as np
import os
import json
import cv2
import scipy.misc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


f1, f2, f3, f4, f5 = 3, 128, 256, 512, 1024
s1, s2, s3, s4, s5 = 64, 32, 16, 8, 4

def load_images():
    # Loading JSON Config
    data = json.load(open("processed.json", "r"))
    print "Loaded Config"

    # Loading Images
    print "----- Loading Images -----"
    images = None
    for emoji in data:
        img = scipy.misc.imread(emoji["location"]).astype(np.float)
        # img = cv2.normalize(img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img/127.5 -1
        if images is None:
            images = np.array([img])
        else:
            images = np.vstack((images, [img]))

    return images

def lrelu(x,alpha=0.1):
	return tf.maximum(x,alpha*x)

def conv2d(x, features, kernel=[5,5], strides=[1,2,2,1], name="conv_layer"):
	with tf.variable_scope(name) as scope:
		weights = weight(shape=kernel + features, name="weights")
		biases = bias(shape=[features[-1]], name="bias")
		output = tf.nn.conv2d(x, weights, strides=strides, padding='SAME') 
		output = tf.nn.bias_add(output, biases)
		return output

def deconv2d(x, features, output_shape, kernel=[5,5], strides=[1,2,2,1], name="deconv_layer"):
	with tf.variable_scope(name) as scope:
		weights = weight(shape=kernel + features, name="weights")
		biases = bias(shape=[features[0]], name="bias")
		output = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape, strides=strides, padding='SAME') 
		return tf.reshape(tf.nn.bias_add(output, biases), output.get_shape())

def bias(shape, name):
	return tf.get_variable(name, shape,initializer=tf.constant_initializer(0.00000))

def weight(shape, name):
	return tf.get_variable(name, shape,initializer=tf.truncated_normal_initializer(stddev=0.02))

def dense(x, shape, name):
	with tf.variable_scope(name):
		weights = weight(shape, name="weights")
		biases = bias([shape[-1]], name="bias")
		return tf.matmul(x,weights) + biases


def batch_norm(inputs, decay=0.9, epsilon=0.00001, scale=True, isTrain=True, name="batch_norm"):
	return tf.contrib.layers.batch_norm(inputs, decay=decay, scale=scale, epsilon=epsilon, updates_collections=None, is_training=isTrain, scope=name)

def discriminator(images, batch_size, reuse):
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()
		# 64x64x3
		output = conv2d(images, features=[f1, f2], name="d_conv_layer_1")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_1")
		output = lrelu(output)

		# 32x32x8
		output = conv2d(output, features=[f2, f3], name="d_conv_layer_2")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_2")
		output = lrelu(output)

		# 16x16x16
		output = conv2d(output, features=[f3, f4], name="d_conv_layer_3")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_3")
		output = lrelu(output)

		# 8x8x32
		output = conv2d(output, features=[f4, f5], name="d_conv_layer_4")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_4")
		output = lrelu(output)

		# 4x4x64
		output = tf.reshape(output, [batch_size, -1])
		output = dense(output, shape=[s5*s5*f5, 1], name="d_dense")
		return output, tf.nn.sigmoid(output)

# def save_sample(sample, sample_size):
# 	x = int(np.sqrt(sample_size))
# 	for i in range(x+1):
# 		images = sample[i*x : (i+1)*x]
# 		for j in range(images.shape[0]):

def sampler(z, batch_size):
	with tf.variable_scope("generator") as scope:
		scope.reuse_variables()
		# 4x4x64
		output = tf.reshape(z, [-1, 1, 1, 30])
		output = deconv2d(output, features=[2048, 30], output_shape=[batch_size,2,2,2048], name="g_deconv_layer_0_1")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_0_1")
		output = tf.nn.relu(output)

		output = deconv2d(output, features=[f5, 2048], output_shape=[batch_size,s5,s5,f5], name="g_deconv_layer_0")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_0")
		output = tf.nn.relu(output)

		output = deconv2d(output, features=[f4, f5], output_shape=[batch_size,s4,s4,f4], name="g_deconv_layer_1")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_1")
		output = tf.nn.relu(output)

		# 8x8x32
		output = deconv2d(output, features=[f3, f4], output_shape=[batch_size,s3,s3,f3], name="g_deconv_layer_2")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_2")
		output = tf.nn.relu(output)

		# 16x16x16
		output = deconv2d(output, features=[f2, f3], output_shape=[batch_size,s2,s2,f2], name="g_deconv_layer_3")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_3")
		output = tf.nn.relu(output)

		# 32x32x8
		output = deconv2d(output, features=[f1, f2], output_shape=[batch_size,s1,s1,f1], name="g_deconv_layer_4")
		output = tf.nn.tanh(output)
		# 64x64x3
		return output

def generator(z, batch_size):
	with tf.variable_scope("generator") as scope:
		# 4x4x64
		output = tf.reshape(z, [-1, 1, 1, 30])
		output = deconv2d(output, features=[2048, 30], output_shape=[batch_size,2,2,2048], name="g_deconv_layer_0_1")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_0_1")
		output = tf.nn.relu(output)

		output = deconv2d(output, features=[f5, 2048], output_shape=[batch_size,s5,s5,f5], name="g_deconv_layer_0")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_0")
		output = tf.nn.relu(output)

		output = deconv2d(output, features=[f4, f5], output_shape=[batch_size,s4,s4,f4], name="g_deconv_layer_1")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_1")
		output = tf.nn.relu(output)

		# 8x8x32
		output = deconv2d(output, features=[f3, f4], output_shape=[batch_size,s3,s3,f3], name="g_deconv_layer_2")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_2")
		output = tf.nn.relu(output)

		# 16x16x16
		output = deconv2d(output, features=[f2, f3], output_shape=[batch_size,s2,s2,f2], name="g_deconv_layer_3")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_3")
		output = tf.nn.relu(output)

		# 32x32x8
		output = deconv2d(output, features=[f1, f2], output_shape=[batch_size,s1,s1,f1], name="g_deconv_layer_4")
		output = tf.nn.tanh(output)
		# 64x64x3
		return output

images = load_images()

BATCH_SIZE = 16

# Create Everything
z = tf.placeholder(tf.float32, shape=[None, 30], name="z")
real_images = tf.placeholder(tf.float32, shape=[None, s1, s1, f1], name="real_input")

fake_images = generator(z, batch_size=BATCH_SIZE)
real_disc_logits, real_disc = discriminator(real_images, batch_size=BATCH_SIZE, reuse=False)
sample = sampler(z,batch_size=5)
fake_disc_logits, fake_disc = discriminator(fake_images, batch_size=BATCH_SIZE, reuse=True)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_disc_logits, labels=tf.ones_like(fake_disc)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc_logits, labels=tf.ones_like(real_disc)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_disc_logits, labels=tf.zeros_like(fake_disc)))


d_loss = d_loss_fake + d_loss_real

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optim = tf.train.AdamOptimizer(0.00002, beta1=0.4).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.4).minimize(g_loss, var_list=g_vars)

n_batches = int(images.shape[0]/BATCH_SIZE)

sample_z = np.random.uniform(-1, 1, size=(5 , 30)).astype(np.float32)
# Lets do this shit
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(1000):
		np.random.shuffle(images)
		for i in range(n_batches):
			batch_noise = np.random.uniform(-1,1,size=(BATCH_SIZE,30)).astype(np.float32)
			_, DLOSS = sess.run([d_optim, d_loss],feed_dict={ real_images : images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], z : batch_noise})

			# Update G network
			_, GLOSS = sess.run([g_optim, g_loss],feed_dict={ z : batch_noise })

			# # Update G network
			# _, GLOSS = sess.run([g_optim, g_loss],feed_dict={ z : batch_noise })

		print "EPOCH ", epoch, DLOSS, GLOSS
		if epoch % 20 == 0:
			fake_image = sess.run(sample, feed_dict={z : sample_z})
			scipy.misc.imsave("./samples/dcgan/{}.png".format(epoch), (fake_image[0] + 1.)/2.)