from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

import skimage

mnist.train._images = skimage.util.random_noise(mnist.train._images, mode='gaussian', var=1)
mnist.test._images = skimage.util.random_noise(mnist.test._images, mode='gaussian', var=1)
mnist.validation._images = skimage.util.random_noise(mnist.validation._images, mode='gaussian', var=1)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#mnist.test._index_in_epoch = 0
#num_test_iterations = 10000 / 50
#num_correct = 0.0
#for i in xrange(num_test_iterations):
#  test_x, test_y = mnist.test.next_batch(50)
#  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(test_y, 1)), tf.float32))
#  acc = accuracy.eval(feed_dict={x: test_x})
#  num_correct += acc

#print("Total accuracy is %f" % (num_correct / num_test_iterations))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
