import tensorflow as tf
import numpy as np
import os
import json
import cv2
import scipy.misc


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_PATH = "./model/condgan/"

f1, f2, f3, f4, f5 = 3, 64, 128, 256, 512
s1, s2, s3, s4, s5 = 64, 32, 16, 8, 4

images = np.load("image_vectors.npy")[()]
descriptions = np.load("new_descriptions.npy")[()]
desc_vectors = np.load("new_desc_vectors.npy")[()]

image_keys = list(images.keys())
master_descs = set(desc_vectors.keys())

BATCH_SIZE = 16
NOISE_SIZE = 100
n_batches = int(len(image_keys)/BATCH_SIZE)

# Generating Averaged Word Vectors
def sample_embeddings(keys):
	embeddings = []
	for i in range(0,len(keys),2):
		embeddings.append((desc_vectors[keys[i]] + desc_vectors[keys[i+1]])/2)
	return np.stack(embeddings)

def get_batch(keys):
    batch_images, true_labels, fake_labels = [], [], []
    for key in keys:
        img = images[key]
        descs = descriptions[key]
        true_label = desc_vectors[descs[0]]
        fake_descs = list(master_descs - set(descs))
        fake_label = desc_vectors[np.random.choice(fake_descs)]
        batch_images.append(img)
        true_labels.append(true_label)
        fake_labels.append(fake_label)
    batch_images = np.stack(batch_images)
    true_labels = np.stack(true_labels)
    fake_labels = np.stack(fake_labels)
    return batch_images, true_labels, fake_labels

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

def discriminator(images, word_embeddings, batch_size, reuse):
	with tf.variable_scope("discriminator") as scope:
		if reuse:
			scope.reuse_variables()
		# 64x64x3
		output = conv2d(images, features=[f1, f2], name="d_conv_layer_1")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_1")
		output = lrelu(output)

		# 32x32x64
		output = conv2d(output, features=[f2, f3], name="d_conv_layer_2")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_2")
		output = lrelu(output)

		# 16x16x128
		output = conv2d(output, features=[f3, f4], name="d_conv_layer_3")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_3")
		output = lrelu(output)

		# 8x8x256
		output = conv2d(output, features=[f4, f5], name="d_conv_layer_4")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_4")
		output = lrelu(output)

		# 4x4x512
		reduced_embeddings = dense(word_embeddings, shape=[300,100], name="d_dense_1")
		reduced_embeddings = lrelu(batch_norm(reduced_embeddings, isTrain=True, name="d_batch_norm_5"))

		reduced_embeddings = tf.expand_dims(reduced_embeddings,1)
		reduced_embeddings = tf.expand_dims(reduced_embeddings,2)
		tiled_embeddings = tf.tile(reduced_embeddings, [1,4,4,1], name='d_tiled_embeddings')

		output = tf.concat([output, tiled_embeddings], 3, name='d_concat')
		output = conv2d(output, features=[f5+100, f5], strides=[1,1,1,1], name="d_conv_layer_5")
		output = batch_norm(output, isTrain=True, name="d_batch_norm_6")
		output = lrelu(output)

		output = tf.reshape(output, [batch_size, -1])

		output = dense(output, [s5*s5*f5, 1], name="d_dense_2")
		return output, tf.nn.sigmoid(output)

def sampler(z, word_embeddings, batch_size):
	with tf.variable_scope("generator") as scope:
		scope.reuse_variables()
		
		reduced_embeddings = dense(word_embeddings, shape=[300, 100], name="g_embeddings_reduce")

		output = tf.concat([z, reduced_embeddings], 1)
		output = dense(output, shape=[NOISE_SIZE + 100, s5*s5*f5], name="g_dense_1")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_0")
		output = tf.nn.relu(output)
		output = tf.reshape(output, [-1, s5, s5, f5])

		# 4x4x512
		output = deconv2d(output, features=[f4, f5], output_shape=[batch_size,s4,s4,f4], name="g_deconv_layer_1")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_1")
		output = tf.nn.relu(output)

		# 8x8x256
		output = deconv2d(output, features=[f3, f4], output_shape=[batch_size,s3,s3,f3], name="g_deconv_layer_2")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_2")
		output = tf.nn.relu(output)

		# 16x16x128
		output = deconv2d(output, features=[f2, f3], output_shape=[batch_size,s2,s2,f2], name="g_deconv_layer_3")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_3")
		output = tf.nn.relu(output)

		# 32x32x64
		output = deconv2d(output, features=[f1, f2], output_shape=[batch_size,s1,s1,f1], name="g_deconv_layer_4")
		output = tf.nn.tanh(output)
		
		# 64x64x3
		return output

def generator(z, word_embeddings, batch_size):
	with tf.variable_scope("generator") as scope:
		reduced_embeddings = dense(word_embeddings, shape=[300, 100], name="g_embeddings_reduce")

		output = tf.concat([z, reduced_embeddings], 1)
		output = dense(output, shape=[NOISE_SIZE + 100, s5*s5*f5], name="g_dense_1")
		output = batch_norm(output, isTrain=False, name="g_batch_norm_0")
		output = tf.nn.relu(output)
		output = tf.reshape(output, [-1, s5, s5, f5])

		# 4x4x512
		output = deconv2d(output, features=[f4, f5], output_shape=[batch_size,s4,s4,f4], name="g_deconv_layer_1")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_1")
		output = tf.nn.relu(output)

		# 8x8x256
		output = deconv2d(output, features=[f3, f4], output_shape=[batch_size,s3,s3,f3], name="g_deconv_layer_2")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_2")
		output = tf.nn.relu(output)

		# 16x16x128
		output = deconv2d(output, features=[f2, f3], output_shape=[batch_size,s2,s2,f2], name="g_deconv_layer_3")
		output = batch_norm(output, isTrain=True, name="g_batch_norm_3")
		output = tf.nn.relu(output)

		# 32x32x64
		output = deconv2d(output, features=[f1, f2], output_shape=[batch_size,s1,s1,f1], name="g_deconv_layer_4")
		output = tf.nn.tanh(output)
		
		# 64x64x3
		return output

# Create Everything
z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE], name="z")
fake_word_embeddings = tf.placeholder(tf.float32, shape=[None, 300], name="fake_word_embeddings")
real_word_embeddings = tf.placeholder(tf.float32, shape=[None, 300], name="real_word_embeddings")
real_images = tf.placeholder(tf.float32, shape=[None, s1, s1, f1], name="real_input")

# Logits
fake_images = generator(z, real_word_embeddings, batch_size=BATCH_SIZE)
real_img_real_label_disc_logits, real_disc_real = discriminator(real_images, real_word_embeddings, batch_size=BATCH_SIZE, reuse=False)
real_img_fake_label_disc_logits, real_disc_fake = discriminator(real_images, fake_word_embeddings, batch_size=BATCH_SIZE, reuse=True)
sample = sampler(z, real_word_embeddings, batch_size=BATCH_SIZE)
fake_disc_logits, fake_disc = discriminator(fake_images, real_word_embeddings, batch_size=BATCH_SIZE, reuse=True)

# Losses
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_disc_logits, labels=tf.zeros_like(fake_disc)+tf.random_uniform(minval=0,maxval=0.3,shape=tf.shape(fake_disc))))
d_loss_real_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_img_real_label_disc_logits, labels=tf.zeros_like(real_disc_real)+tf.random_uniform(minval=0,maxval=0.3,shape=tf.shape(real_disc_real))))
d_loss_real_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_img_fake_label_disc_logits, labels=tf.ones_like(real_disc_fake)-tf.random_uniform(minval=0,maxval=0.3,shape=tf.shape(real_disc_fake))))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_disc_logits, labels=tf.ones_like(fake_disc)-tf.random_uniform(minval=0,maxval=0.3,shape=tf.shape(fake_disc))))

d_loss = d_loss_fake + (d_loss_real_real + d_loss_real_fake)/2

# Summaries
gloss_sum = tf.summary.scalar('Generator_Loss', g_loss)
dloss_sum = tf.summary.scalar('Discriminator_Loss', d_loss)
dloss_real_real_sum = tf.summary.scalar('Discriminator_Real_I_Real_L_Loss', d_loss_real_real)
dloss_real_fake_sum = tf.summary.scalar('Discriminator_Real_I_Fake_L_Loss', d_loss_real_fake)
dloss_fake_sum = tf.summary.scalar('Discriminator_Fake_I_Real_L_Loss', d_loss_fake)
g_image_sum = tf.summary.image('Generated_Images', fake_images)
merged_scalar = tf.summary.merge([gloss_sum, dloss_sum, dloss_real_real_sum, dloss_real_fake_sum, dloss_fake_sum])
merged_all = tf.summary.merge_all()

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.4).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(0.002, beta1=0.6).minimize(g_loss, var_list=g_vars)

saver = tf.train.Saver(t_vars)

# Lets do this shit
with tf.Session() as sess:
	scalar_writer = tf.summary.FileWriter('./Graph', sess.graph)

	sess.run(tf.global_variables_initializer())

	# Load latest checkpoint
	if tf.train.latest_checkpoint(MODEL_PATH):
		saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

	for epoch in range(50000):
		np.random.shuffle(image_keys)
		dl, gl = [], []
		for i in range(n_batches):
			batch_noise = np.random.uniform(-1,1,size=(BATCH_SIZE, NOISE_SIZE)).astype(np.float32)
			batch_images, batch_true_labels, batch_fake_labels = get_batch(image_keys[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
			_, DLOSS = sess.run([d_optim, d_loss],feed_dict={ real_images : batch_images,real_word_embeddings : batch_true_labels, fake_word_embeddings : batch_fake_labels, z : batch_noise})

			# Update G network
			_, GLOSS = sess.run([g_optim, g_loss],feed_dict={ z : batch_noise, real_word_embeddings : batch_true_labels})

			# Update G network
			_, GLOSS = sess.run([g_optim, g_loss],feed_dict={ z : batch_noise, real_word_embeddings : batch_true_labels})

			dl.append(DLOSS)
			gl.append(GLOSS)
		print "EPOCH ", epoch, np.mean(dl), np.mean(gl)

		# Model Epoch Summary
		if epoch % 500 != 0:
			np.random.shuffle(image_keys)
			batch_noise = np.random.uniform(-1,1,size=(BATCH_SIZE, NOISE_SIZE)).astype(np.float32)
			batch_images, batch_true_labels, batch_fake_labels = get_batch(image_keys[0 : BATCH_SIZE])
			_, _, _, _, _, summary = sess.run([g_loss, d_loss, d_loss_real_real, d_loss_real_fake, d_loss_fake,merged_scalar], feed_dict={ real_images : batch_images, real_word_embeddings : batch_true_labels, fake_word_embeddings : batch_fake_labels, z : batch_noise})
			scalar_writer.add_summary(summary, epoch)

		# Generating Samples using Combined Vectors of two different emotions
		if epoch % 100 == 0:
			sample_z = np.random.uniform(-1,1,size=(BATCH_SIZE, NOISE_SIZE)).astype(np.float32)
			shuffle_descs = list(master_descs)
			np.random.shuffle(shuffle_descs)
			random_labels = shuffle_descs[0:BATCH_SIZE*2]
			fake_image = sess.run(sample, feed_dict={z : sample_z, real_word_embeddings: sample_embeddings(random_labels)})
			if not os.path.isdir("./samples/condgan/"+str(epoch)):
				os.mkdir("./samples/condgan/"+str(epoch))
			else:
				for f in os.listdir("./samples/condgan/"+str(epoch)):
					os.remove(os.path.join("./samples/condgan/"+str(epoch),f))
			for j in range(BATCH_SIZE):
				scipy.misc.imsave("./samples/condgan/{}/{}.png".format(epoch, random_labels[j*2]+"_"+random_labels[j*2+1]), (fake_image[j] + 1.)/2.)

		# Model Checkpoint
		if epoch % 500 == 0:
			# Summary with Images
			np.random.shuffle(image_keys)
			batch_noise = np.random.uniform(-1,1,size=(BATCH_SIZE, NOISE_SIZE)).astype(np.float32)
			batch_images, batch_true_labels, batch_fake_labels = get_batch(image_keys[0 : BATCH_SIZE])
			_, _, _, _, _, _, summary = sess.run([g_loss, d_loss, d_loss_real_real, d_loss_real_fake, d_loss_fake, fake_images, merged_all],feed_dict={ real_images : batch_images, real_word_embeddings : batch_true_labels, fake_word_embeddings : batch_fake_labels, z : batch_noise})
			scalar_writer.add_summary(summary, epoch)

			path = saver.save(sess, MODEL_PATH, global_step=epoch)
			print "Checkpoint Saved at {}".format(path)