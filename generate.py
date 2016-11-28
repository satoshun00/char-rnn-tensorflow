import tensorflow as tf
from tensorflow.python.ops import rnn_cell

import argparse
import time
import os
import pickle
import numpy as np

import data_utils
import charrnn

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', type=int, default=40,
                     help='number of words to sample')
  parser.add_argument('--prime', type=str, default=u'The',
                     help='prime text')

  args, _ = parser.parse_known_args()
  generate(args)

def generate(args):
  with open(os.path.join('save', 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)
  # with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
  #   chars, vocab = pickle.load(f)

  num = args.n
  prime = args.prime
  batch_size = 1
  seq_length = 1
  rnn_size = saved_args.rnn_size
  num_layers = saved_args.num_layers
  vocab_size = saved_args.vocabulary_size

  vocab_path = os.path.join('data/development', "vocab%d.txt" % vocab_size)
  vocab, vocab_rev = data_utils.initialize_vocabulary(vocab_path)

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
  ckpt = tf.train.get_checkpoint_state('save')
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    state = sess.run(initial_state)

    # for word in [data_utils._GO] + prime.split(' '):
    words = data_utils.sentence_to_token_ids(tf.compat.as_bytes(data_utils._GO.decode('utf8') + ' ' + prime), vocab)
    for word in words:
      x = np.zeros((1, 1))
      x[0, 0] = word
      feed = {input_placeholder: x, initial_state: state}
      [state] = sess.run([last_state], feed)

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    sent = prime
    word = words[-1]
    for n in range(num):
      x = np.zeros((1, 1))
      x[0, 0] = word
      feed = {input_placeholder: x, initial_state: state}
      [probs, state] = sess.run([probs_op, last_state], feed)
      p = probs[0]
      sample = weighted_pick(p)

      pred = vocab_rev[sample]
      sent += pred + ' '
      word = sample
      if (pred == data_utils._EOS.decode('utf8')):
        break
    print(sent)

if __name__ == '__main__':
  main()
