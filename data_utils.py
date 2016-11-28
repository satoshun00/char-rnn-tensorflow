import gzip
import os
import re
import tarfile
import numpy as np

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w.decode(), UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w).decode(), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, normalize_digits=True):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          line = tf.compat.as_bytes(line)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare(data_dir, vocabulary_size, tokenizer=None):
  path = os.path.join(data_dir, "development/newstest2013.txt")

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "development/vocab%d.txt" % vocabulary_size)
  create_vocabulary(vocab_path, path, vocabulary_size, tokenizer)

  # Create token ids for the training data.
  ids_path = path + (".ids%d" % vocabulary_size)
  data_to_token_ids(path, ids_path, vocab_path, tokenizer)

  return (ids_path, vocab_path)

def read_data(input_path, batch_size):
  data_set = []
  max_seq_length = 0
  with tf.gfile.GFile(input_path, mode="r") as input_file:
    inputs = input_file.readline()
    counter = 0
    while inputs:
      counter += 1
      if counter % 100000 == 0:
        print("  reading data line %d" % counter)
        sys.stdout.flush()
      ids = [int(x) for x in inputs.split()]
      source_ids = [GO_ID] + ids
      target_ids = ids + [EOS_ID]
      data_set.append([source_ids, target_ids])
      max_seq_length = max(max_seq_length, len(ids) + 1)
      inputs = input_file.readline()
  num_batches = int(len(data_set) / batch_size)
  data_set = data_set[:num_batches * batch_size]
  return (data_set, num_batches, max_seq_length)

def get_batch(data_set, batch_idx, batch_size, seq_length):
  x_list, y_list = [], []

  batch_data = data_set[batch_idx * batch_size:batch_idx * batch_size + batch_size]
  for i in range(batch_size):
    x, y = batch_data[i]

    x_pad = [PAD_ID] * (seq_length - len(x))
    x_list.append(x + x_pad)

    y_pad = [PAD_ID] * (seq_length - len(y))
    y_list.append(y + y_pad)

  batch_x, batch_y = [], []

  return (np.array(x_list, dtype=np.int32), np.array(y_list, dtype=np.int32))
