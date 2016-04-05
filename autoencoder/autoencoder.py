"""Tutorial on how to create an autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
# %% Imports
import tensorflow as tf
import numpy as np
import math

b_size = 50
def fully_connected(x, b, W):
  (height, width) = W.get_shape().as_list()
  x_1 = tf.reshape(x, [-1, height, 1])
  x_2 = tf.tile(x_1, [1, 1, width])
  x_3 = tf.add(x_2, b)
  W_t = tf.transpose(W)
  x_hat = tf.nn.relu(x_3)
  list_result = []
  for i in xrange(b_size):
    matmul_result = tf.matmul(W_t, tf.reshape(tf.slice(x_hat, [i, 0, 0], [1, -1, -1]), [height, width]))
    list_result.append(tf.gather(tf.reshape(matmul_result, [-1]), [j*width+j for j in xrange(width)]))

  return tf.pack(list_result)

# %% Autoencoder definition
def autoencoder(dimensions=[784, 512, 256, 64]):
    """Build a deep autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %% input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b1 = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))

        b2 = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(fully_connected(current_input, b1, W) + b2)
        current_input = output

    # %% latent representation
    z = current_input
    encoder.reverse()

    # %% Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        n_input = int(current_input.get_shape()[1])
        W = tf.transpose(encoder[layer_i])
        (W_h, W_w) = W.get_shape().as_list()
        b1 = tf.Variable(
            tf.random_uniform([W_h, W_w],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))


        b2 = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(fully_connected(current_input, b1, W) + b2)
        current_input = output

    # %% now have the reconstruction through the network
    y = current_input

    # %% cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}

# %% Basic test
def test_mnist():
    """Test the autoencoder using MNIST."""
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(dimensions=[784, 256, 64])

    # %%
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # %%
    # Fit all training data
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // b_size):
            batch_xs, _ = mnist.train.next_batch(b_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    fig.show()
    plt.draw()
    fig.waitforbuttonpress()

# %%
if __name__ == '__main__':
    test_mnist()
