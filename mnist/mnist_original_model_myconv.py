import math
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

b_size = 1

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, strides=[1, 1, 1, 1]):
  # Set convolution parameters
  (filter_height, filter_width, in_depth, out_depth) = W.get_shape().as_list()
  (batch_size, in_height, in_width, input_channels) = x.get_shape().as_list()
  out_height = int(math.ceil(float(in_height) / float(strides[1])))
  out_width = int(math.ceil(float(in_width) / float(strides[2])))
  pad_along_height = ((out_height - 1) * strides[1] + filter_height - in_height)
  pad_along_width = ((out_width - 1) * strides[2] + filter_width - in_width)
  pad_top = int(pad_along_height / 2)
  # If height padding is odd, add leftover pixel to bottom
  pad_bottom = int((pad_along_height + 1) / 2)
  pad_left = int(pad_along_width / 2)
  pad_right = int((pad_along_width + 1) / 2)

  # Add padding to input tensor
  padded_x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
  
  # Tensor iteration
  out_tensor_arr = [[[[tf.constant([-1])]*out_depth]*out_width]*out_height]*b_size
  for batch_idx in xrange(b_size):
    for in_idx in xrange(in_depth):
      for i in xrange(in_height - filter_height + 1):
        for j in xrange(in_width - filter_width + 1):
          input_slice = tf.slice(x, [batch_idx, i, j, in_idx], [1, filter_height, filter_width, 1])
          input_2d = tf.squeeze(input_slice)
          input_2d = tf.reshape(input_2d, [filter_height * filter_width, 1])
          for out_idx in xrange(out_depth):
            # Calculate 
            weight_slice = tf.slice(W, [0, 0, in_idx, out_idx], [-1, -1, 1, 1])
            weight_2d = tf.squeeze(weight_slice)
            weight_2d = tf.reshape(weight_2d, [1, filter_height * filter_width])
            prod = tf.matmul(weight_2d, input_2d)
            out_tensor_arr[batch_idx][i / strides[1]][j / strides[2]][out_idx] = \
              tf.reshape(prod, [-1])
 
  print("Gonna map") 
  def concat_inner_tensors(tensor_list):
    return tf.concat(0, tensor_list)
  
  y = map(lambda a : map(lambda b : map(concat_inner_tensors, b), a), out_tensor_arr)
  print(y[0][0][0])
  z = map(lambda a : map (tf.pack, a), y)
  print(z[0][0])
  z1 = map(tf.pack, z)
  print(z1[0])
  res = tf.pack(z1)
  print(res)
  return res
  #output_tensor = tf.zeros([b_size, out_height, out_width, out_depth], tf.float32)
  #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME)
  
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

conv2d(x_image, W_conv1, [1, 1, 1, 1])

print("First conv ran")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print("First Conv finished")
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

print("Second conv about to run")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print("Second conv finished")
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
  batch = mnist.train.next_batch(b_size)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
