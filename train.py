import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import argparse
import time
import os
import pickle

import charrnn
import data_utils
# from utils import TextLoader

def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data',
                     help='data directory containing input.txt')
  parser.add_argument('--save_dir', type=str, default='save',
                     help='directory to store checkpointed models')
  parser.add_argument('--rnn_size', type=int, default=128,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=2,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=60,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=20,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=1000,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=5.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.002,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.97,
                     help='decay rate for rmsprop')
  parser.add_argument('--vocabulary_size', type=int, default=2000,
                     help='number of vocabulary')
  parser.add_argument('--init_from', type=str, default=None,
                     help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                          'config.pkl'        : configuration;
                          'chars_vocab.pkl'   : vocabulary definitions;
                          'checkpoint'        : paths to model file(s) (created by tf).
                                                Note: this file contains absolute paths, be careful when moving files around;
                          'model.ckpt-*'      : file(s) with model definition (created by tf)
                      """)
  args, _ = parser.parse_known_args()
  run_training(args)

def run_training(args):
  data_dir = args.data_dir
  batch_size = args.batch_size
  # seq_length = args.seq_length
  rnn_size = args.rnn_size
  num_layers = args.num_layers
  grad_clip = args.grad_clip
  num_epochs = args.num_epochs
  initial_learning_rate = args.learning_rate
  decay_rate = args.decay_rate
  init_from = args.init_from
  save_every = args.save_every
  save_dir = args.save_dir
  vocab_size = args.vocabulary_size

  with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(args, f)

  ids_path, _ = data_utils.prepare(data_dir, vocab_size)
  # data_loader = TextLoader(data_dir, batch_size, seq_length)

  data_set, num_batches, seq_length = data_utils.read_data(ids_path, batch_size)

  # vocab_size = data_loader.vocab_size

  input_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])
  targets_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])

  cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
  cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

  initial_state = cell.zero_state(batch_size, tf.float32)

  logits, last_state = charrnn.inference(input_placeholder, initial_state, cell, rnn_size, vocab_size, seq_length, False)

  loss = charrnn.loss(logits, targets_placeholder, batch_size, seq_length, vocab_size)

  learning_rate = tf.Variable(0.0, trainable=False)

  train_op, cost = charrnn.training(loss, learning_rate, batch_size, seq_length, grad_clip)

  init = tf.initialize_all_variables()

  saver = tf.train.Saver(tf.all_variables())

  sess = tf.Session()

  sess.run(init)

  # print(data_utils.get_batch(data_set, 0, batch_size, seq_length))
  for e in range(num_epochs):
    sess.run(tf.assign(learning_rate, initial_learning_rate * (decay_rate ** e)))
    state = sess.run(initial_state)
    # data_loader.reset_batch_pointer()
    for batch_idx in range(num_batches):
      start = time.time()
      x, y = data_utils.get_batch(data_set, batch_idx, batch_size, seq_length)
      # x, y = data_loader.next_batch()
      feed = {input_placeholder: x, targets_placeholder: y}
      for i, (c, h) in enumerate(initial_state):
        feed[c] = state[i].c
        feed[h] = state[i].h
      train_loss, state, _ = sess.run([cost, last_state, train_op], feed)
      end = time.time()
      print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
        .format(e * num_batches + batch_idx,
                num_epochs * num_batches,
                e, train_loss, end - start))
      if (e * num_batches + batch_idx) % save_every == 0\
          or (e==num_epochs-1 and batch_idx == num_batches-1): # save for the last result
          checkpoint_path = os.path.join(save_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step = e * num_batches + batch_idx)
          print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
  tf.app.run(main)
