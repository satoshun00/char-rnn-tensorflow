import tensorflow as tf
from tensorflow.python.ops import rnn_cell

import argparse
import time
import os
import pickle
import numpy as np

import charrnn

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default='save',
                     help='model directory to store checkpointed models')
  parser.add_argument('-n', type=int, default=500,
                     help='number of characters to sample')
  parser.add_argument('--prime', type=str, default=u' ',
                     help='prime text')
  parser.add_argument('--sample', type=int, default=1,
                     help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

  args, _ = parser.parse_known_args()
  generate(args)

def generate(args):
  with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)
  with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
    chars, vocab = pickle.load(f)

  num = args.n
  prime = args.prime
  batch_size = 1
  seq_length = 1
  rnn_size = saved_args.rnn_size
  num_layers = saved_args.num_layers
  vocab_size = saved_args.vocab_size

  input_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])
  targets_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])

  cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
  cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

  initial_state = cell.zero_state(batch_size, tf.float32)

  logits, last_state = charrnn.inference(input_placeholder, initial_state, cell, rnn_size, vocab_size, seq_length, True)

  probs_op = charrnn.evaluation(logits)

  init = tf.initialize_all_variables()
  sess = tf.Session()

  sess.run(init)
  saver = tf.train.Saver(tf.all_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    state = sess.run(initial_state)

    for char in prime[:-1]:
      x = np.zeros((1, 1))
      x[0, 0] = vocab[char]
      feed = {input_placeholder: x, initial_state: state}
      [state] = sess.run([last_state], feed)

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    sent = prime
    char = prime[-1]
    for n in range(num):
      x = np.zeros((1, 1))
      x[0, 0] = vocab[char]
      feed = {input_placeholder: x, initial_state: state}
      [probs, state] = sess.run([probs_op, last_state], feed)
      p = probs[0]

      sample = weighted_pick(p)

      pred = chars[sample]
      sent += pred
      char = pred
    print(sent)

if __name__ == '__main__':
  main()
