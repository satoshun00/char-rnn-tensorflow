import argparse
import os.path
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import mnist

# Basic model parameters as external flags.
FLAGS = None

LOG_DIR = '/tmp/tensorflow/mnist/logs/fully_connected_feed'
INPUT_DATA_DIR = '/tmp/tensorflow/mnist/input_data'
BATCH_SIZE = 100
HIDDEN1 = 128
HIDDEN2 = 32
LEARNING_RATE = 0.01
MAX_STEPS = 4000

def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
  images_feed, labels_feed = data_set.next_batch(BATCH_SIZE, FLAGS.fake_data)
  feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
  }
  return feed_dict

def do_evaluation(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
  true_count = 0
  steps_per_epoch = data_set.num_examples // BATCH_SIZE
  num_examples = steps_per_epoch * BATCH_SIZE
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

def run_training():
  data_sets = input_data.read_data_sets(INPUT_DATA_DIR, FLAGS.fake_data)
  with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

    logits = mnist.inference(images_placeholder, HIDDEN1, HIDDEN2)

    loss = mnist.loss(logits, labels_placeholder)

    train_op = mnist.training(loss, LEARNING_RATE)

    eval_correct = mnist.evaluation(logits, labels_placeholder)

    summary = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    # saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph)

    sess.run(init)

    for step in range(MAX_STEPS):
      start_time = time.time()

      feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
        checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
        # saver.save(sess, checkpoint_file, global_step=step)
        print('Training Data Evaluation:')
        do_evaluation(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
        print('Validation Data Evaluation:')
        do_evaluation(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
        print('Test Data Evaluation:')
        do_evaluation(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def main(_):
  if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
  tf.gfile.MakeDirs(LOG_DIR)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main)
