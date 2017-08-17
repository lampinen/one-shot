# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Train the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import os
from collections import Counter

import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("train_file_path", None,
                    "Used to overwrite the path to the train_file.")
flags.DEFINE_string("word_train_file_path", None,
                    "Path to train file for word.")
flags.DEFINE_string("word_test_file_path", None,
                    "Path to train file for word.")
flags.DEFINE_string("new_word", None,
                    "New word to learn.")
flags.DEFINE_string("vocab_file_path", "raw_data/ptb.train.txt",
                    "File from which to build the model vocabulary.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("result_log_file", "results_log.csv",
                    "File in save_path directory to write results log to.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_word_train_sentences", 1,
		 "Number of training sentences given for word")
flags.DEFINE_bool("reload_pre", False, "Reload pre-trained network.")
flags.DEFINE_bool("centroid_approach", False, "Just set new word vector to be centroid of all the other words in the sentence.")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, new_word_index):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      self.embedding = embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(
    #     cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    # use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )

    # update the cost variables
    self._cost = cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

    # Optimizer and ops for finding best representation for a new word:
    word_selection_onehot = tf.expand_dims(tf.one_hot(new_word_index, vocab_size), axis=1)
    embedding_var = [v for v in tf.trainable_variables() if "embedding" in v.name]
    embedding_optimizer = tf.train.GradientDescentOptimizer(self._lr) 
    word_grads_and_vars = embedding_optimizer.compute_gradients(
	loss, var_list=embedding_var)
    # Only update target word embedding
    masked_word_grads_and_vars = [(tf.multiply(x[0], word_selection_onehot), x[1]) for x in word_grads_and_vars]
    self.word_train_op = embedding_optimizer.apply_gradients(masked_word_grads_and_vars)

    # Assign op and ph for centroid_approach
    if FLAGS.centroid_approach:
      self.embedding_assign_ph = tf.placeholder(tf.float32, shape=embedding.get_shape())
      self.embedding_assign_op = tf.assign(embedding, self.embedding_assign_ph)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
#  max_wordopt_epoch = 100
#  wordopt_lr = 0.01
#  wordopt_lr_decay = 0.95
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
#  max_wordopt_epoch = 100
#  wordopt_lr = 0.01
#  wordopt_lr_decay = 0.95
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  max_wordopt_epoch = 50
  wordopt_lr = 0.01
  wordopt_lr_decay = 0.95
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  max_wordopt_epoch = 10
  keep_prob = 1.0
  lr_decay = 0.5
  wordopt_lr = 0.01
  wordopt_lr_decay = 0.99
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path,
				 FLAGS.train_file_path,
				 FLAGS.vocab_file_path,
				 FLAGS.word_train_file_path,
				 FLAGS.word_test_file_path)
  train_data, valid_data, test_data, word_train_data, word_test_data, vocabulary = raw_data

  if not FLAGS.new_word:
    raise ValueError("Must set --new_word to the new word to be learned")
  new_word_index = vocabulary[FLAGS.new_word]

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  word_train_config = get_config()
  word_train_config.batch_size = FLAGS.num_word_train_sentences
  word_train_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input, new_word_index=new_word_index)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input, new_word_index=new_word_index)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input, new_word_index=new_word_index)

    with tf.name_scope("WordOptTrain"):
      word_train_input = PTBInput(config=word_train_config, data=word_train_data, name="WordOptInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mwordtrain = PTBModel(is_training=True, config=word_train_config, input_=word_train_input, new_word_index=new_word_index)
      tf.summary.scalar("Word Train Loss", mwordtrain.cost)
      tf.summary.scalar("Learning Rate", mwordtrain.lr)

    with tf.name_scope("WordOptTest"):
      word_test_input = PTBInput(config=eval_config, data=word_test_data, name="WordOptTestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mwordtest = PTBModel(is_training=False, config=eval_config,
                         input_=word_test_input, new_word_index=new_word_index)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      if not FLAGS.reload_pre:
	for i in range(config.max_max_epoch):
	  lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
	  m.assign_lr(session, config.learning_rate * lr_decay)

	  print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
	  train_perplexity = run_epoch(session, m, eval_op=m.train_op,
	      			 verbose=True)
	  print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
	  valid_perplexity = run_epoch(session, mvalid)
	  print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

	if FLAGS.save_path:
	  curr_save_path = FLAGS.save_path + "/" + FLAGS.new_word +  "/pre_fine/"
	  if not os.path.isdir(curr_save_path):
	      os.makedirs(curr_save_path)
	  print("Saving model to %s." % curr_save_path)
	  sv.saver.save(session, curr_save_path, global_step=sv.global_step)

      else:  # FLAGS.reload_pre
	curr_save_path = FLAGS.save_path + "/" + FLAGS.new_word +  "/pre_fine/"
	if not os.path.isdir(curr_save_path):
	    os.makedirs(curr_save_path)
	print("Loading model from %s." % curr_save_path)
	sv.saver.restore(session, tf.train.latest_checkpoint(curr_save_path))
	print("Successfully restored model.")

      word_test_perplexity = run_epoch(session, mwordtest)
      print("Word Test Perplexity: %.3f" % (word_test_perplexity))
      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)
      if FLAGS.save_path:
	with open(FLAGS.save_path + "/" + FLAGS.new_word + "/" + FLAGS.result_log_file, "a") as flog:
	  flog.write("pre_new_word_test_perp, %f\n" %(word_test_perplexity))
	  flog.write("pre_test_perp, %f\n" %(test_perplexity))

      # Optimize for new word.
      if not FLAGS.centroid_approach:
	for i in range(config.max_wordopt_epoch):
  #	blah = session.run(mwordtest.embedding) 
  #	print(blah[new_word_index])
	  lr_decay = config.wordopt_lr_decay ** max(i + 1 - config.max_epoch, 0.0)
	  mwordtrain.assign_lr(session, config.wordopt_lr * lr_decay)
	  print("Word Opt Epoch: %d Learning rate: %f" % (i + 1, session.run(mwordtrain.lr)))
	  word_train_perplexity = run_epoch(session, mwordtrain, eval_op=mwordtrain.word_train_op,
				       verbose=True)
	  print("Word Opt Epoch: %d Word Train Perplexity: %.3f" % (i + 1, word_train_perplexity))
	  word_test_perplexity = run_epoch(session, mwordtest)
	  print("Word Opt Epoch: %d Word Test Perplexity: %.3f" % (i + 1, word_test_perplexity))

	if FLAGS.save_path:
	  curr_save_path = FLAGS.save_path + "/" + FLAGS.new_word +  "/post_fine/"
	  if not os.path.isdir(curr_save_path):
	      os.makedirs(curr_save_path)
	  print("Saving model to %s." % curr_save_path)
	  sv.saver.save(session, curr_save_path, global_step=sv.global_step)
      else:  # FLAGS.centroid_approach
	curr_embedding = session.run(mwordtest.embedding)
	word_counts = Counter()
	with open(FLAGS.word_train_file_path, "r") as fin:
	  for line in fin.readlines():
	    word_counts.update(line.split())
	del(word_counts[FLAGS.new_word])
	new_word_embedding = np.zeros_like(curr_embedding[new_word_index])
	
	for word in word_counts:
	  new_word_embedding += curr_embedding[vocabulary[word]] * word_counts[word]
	new_word_embedding /= sum(word_counts.values())
	print(curr_embedding[new_word_index])
	curr_embedding[new_word_index] = new_word_embedding 
	print(curr_embedding[new_word_index])	
	session.run(mwordtrain.embedding_assign_op, feed_dict={mwordtrain.embedding_assign_ph: curr_embedding})
	print("Centroid-based embedding for new word assigned")
	
      word_test_perplexity = run_epoch(session, mwordtest)
      print("Word Test Perplexity: %.3f" % (word_test_perplexity))
      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)
      if FLAGS.save_path:
	with open(FLAGS.save_path + "/" + FLAGS.new_word + "/" + FLAGS.result_log_file, "a") as flog:
	  flog.write("post_new_word_test_perp, %f\n" %(word_test_perplexity))
	  flog.write("post_test_perp, %f\n" %(test_perplexity))

      


if __name__ == "__main__":
  tf.app.run()
