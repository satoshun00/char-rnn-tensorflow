import argparse
import sys
import time
# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

FLAGS = None

def main(_):
  # download data
  mnist_data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
  W = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES]))
  b = tf.Variable(tf.zeros([NUM_CLASSES]))
  y = tf.matmul(x, W) + b

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  start = time.time()
  for _ in range(1000):
    batch_xs, batch_ys = mnist_data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist_data.test.images,
                                      y_: mnist_data.test.labels}))
  print(("end:{0}".format(time.time() - start)) + "[sec]")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                    help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main)
