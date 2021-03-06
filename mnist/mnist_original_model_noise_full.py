from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import skimage

b_size = 50
mnist.train._images = skimage.util.random_noise(mnist.train._images, mode='gaussian', var=1)
mnist.test._images = skimage.util.random_noise(mnist.test._images, mode='gaussian', var=1)
mnist.validation._images = skimage.util.random_noise(mnist.validation._images, mode='gaussian', var=1)

print(mnist.train._images[0])

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([784, 10]))
b2 = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

def fully_connected(x, b, W):
  (height, width) = W.get_shape().as_list()
  batch_size = tf.shape(x)[0]
  x_1 = tf.reshape(x, [-1, height, 1])
  x_2 = tf.tile(x_1, [1, 1, width])
  x_3 = tf.add(x_2, b)
  W_t = tf.transpose(W)
  x_hat = tf.nn.relu(x_3)
  list_result = []
  #batch_size = tf.reshape(tf.shape(x)[0])
  for i in xrange(b_size):
    matmul_result = tf.matmul(W_t, tf.reshape(tf.slice(x_hat, [i, 0, 0], [1, -1, -1]), [height, width]))
    list_result.append(tf.gather(tf.reshape(matmul_result, [-1]), [j*width+j for j in xrange(width)]))

  return tf.pack(list_result)

x_c = fully_connected(x, b, W) + b2
y = tf.nn.softmax(x_c)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(b_size)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#num_test_iterations = 1# 1000 / b_size
#test_input_index = 0
#for i in xrange(num_test_iterations):
#  test_x, test_y = mnist.test.next_batch(b_size)
#  accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(test_y, 1))
#  print(accuracy.eval(feed_dict={x: test_x}))
mnist.test._index_in_epoch = 0
num_test_iterations = 10000 / 50
num_correct = 0.0
for i in xrange(num_test_iterations):
  test_x, test_y = mnist.test.next_batch(50)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(test_y, 1)), tf.float32))
  acc = accuracy.eval(feed_dict={x: test_x})
  num_correct += acc

print("Total accuracy is %f" % (num_correct / num_test_iterations))
#print(y.eval(feed_dict={x: mnist.test.images}))
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

'''def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))'''

