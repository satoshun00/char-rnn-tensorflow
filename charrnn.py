import tensorflow as tf
from tensorflow.python.ops import seq2seq

def inference(input_placeholder, state, cell, rnn_size, vocab_size, seq_length, predict):
  with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
      inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_placeholder))
      inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

  def loop(prev, _):
    prev = tf.matmul(prev, softmax_w) + softmax_b
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    return tf.nn.embedding_lookup(embedding, prev_symbol)

  outputs, last_state = seq2seq.rnn_decoder(inputs, state, cell, loop_function=loop if predict else None, scope='rnnlm')
  output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
  logits = tf.matmul(output, softmax_w) + softmax_b
  return [logits, last_state]

def loss(logits, targets_placeholder, batch_size, seq_length, vocab_size):
  loss = seq2seq.sequence_loss_by_example([logits],
          [tf.reshape(targets_placeholder, [-1])],
          [tf.ones([batch_size * seq_length])],
          vocab_size)
  return loss

def training(loss, learning_rate, batch_size, seq_length, grad_clip):
  cost = tf.reduce_sum(loss) / batch_size / seq_length
  
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
          grad_clip)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(zip(grads, tvars))
  return [train_op, cost]

def evaluation(logits):
  return tf.nn.softmax(logits)
