import pandas as pd
import numpy as np
import re
import jieba
jieba.set_dictionary('datas/dict/dict.txt.big')
jieba.load_userdict('datas/dict/edu_dict.txt')

# Load word2vec model
from gensim.models import word2vec
word2vec_model = word2vec.Word2Vec.load('models/word2vec_250.model.bin')

# Save the vocab
# Sequences have various lengths, so let index '0' serve as padding  -> all index+1
word2id = dict([(k, v.index+1) for k, v in word2vec_model.wv.vocab.items()])
word2vec_vocab_size = len(word2id)
id2word = dict([(v, k) for k, v in word2id.items()])

# Release unused memory comsumed model
import time
import gc
del(word2vec_model)
time.sleep(1)
gc.collect()


'''  Read in official sample data '''
sample = pd.read_csv('datas/sample_test_data.txt')
# Extract sample test datas
x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.dialogue.values]
x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.options.values]
# y = sample.answer.values

# Tokenize
x1 = np.array([list(jieba.cut(' '.join(_))) for _ in x1])
x2 = np.array([[list(jieba.cut(s)) for s in _] for _ in x2])
assert(np.sum([len(_)!=6 for _ in x2]) == 0)


''' Convert string list x1, x2 to np array of index '''
# Find the length of longest sequence, we shall pad all sentences to this length
max_seq_len = 0
for x in x1:
    max_seq_len = max(max_seq_len, len(x))
for xs in x2:
    for x in xs:
        max_seq_len = max(max_seq_len, len(x))

new_x1 = []
for sentence in x1:
    tmp_sentence = []
    # Converd word to index
    for word in sentence:
        if word in word2id:
            tmp_sentence.append(word2id[word])
    
    # Padding all sequences to same length
    len_to_pad = max_seq_len - len(tmp_sentence)
    tmp_sentence.extend([0] * len_to_pad)
    new_x1.append(tmp_sentence)    
x1 = np.array(new_x1)

new_x2 = []
for options in x2:
    for sentence in options:
        tmp_sentence = []
        for word in sentence:
            if word in word2id:
                tmp_sentence.append(word2id[word])

        # Padding all sequences to same length
        len_to_pad = max_seq_len - len(tmp_sentence)
        tmp_sentence.extend([0] * len_to_pad)
        new_x2.append(tmp_sentence)
    
x2 = np.array(new_x2)


# Repeate x1  -> (x1[0], x1[0], x1[0], x1[0], x1[0], x1[0],  x1[1], ...)
num_responses = 6
x1 = np.repeat(x1, num_responses, axis=0)

# Original 'y' means which response is correct
y = sample.answer.values
# Now convert y to indicate wherther one (context, respoonse) is corrct, 0/1
new_y = []
for answer in y:
    new_y.extend([0]*answer)
    new_y.append(1)
    new_y.extend([0]*(num_responses-answer-1))
y = np.reshape(np.array(new_y), (-1, 1))
print(y.shape)


''' Parameters '''
# Define hyperparameters
# reference: https://github.com/dennybritz/chatbot-retrieval/blob/8b1be4c2e63631b1180b97ef927dc2c1f7fe9bea/udc_hparams.py
# Model Parameters
params = {}
params['word2vec_path'] = 'models/word2vec_250.model.bin.wv.syn0.npy'
params['word2vec_vocab_size'] = word2vec_vocab_size
params['word2vec_dim'] = 250
params['rnn_dim'] = 256

# Training Parameters
params['learning_rate'] = 0.001
params['batch_size'] = 10 #128
params['eval_batch_size'] = 6  #16
params['n_iterations'] = 5


''' Self-defined Batch helper '''
def shuffle(*pairs):
    pairs = list(pairs)
    for i, pair in enumerate(pairs):
        pairs[i] = np.array(pair)
    p = np.random.permutation(len(pairs[0]))
    return tuple(pair[p] for pair in pairs)


class Dataset():
    def __init__(self, X1, X2, Y, batch_size):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.X1, self.X2, self.Y = shuffle(self.X1, self.X2, self.Y)
        self.batch_size = batch_size
        self.state = 0
        self.total_batch = int(X1.shape[0] / batch_size)

    def next_batch(self):
        start = self.state * self.batch_size
        end = start + self.batch_size
        next_x1 = self.X1[start:end]
        next_x2 = self.X2[start:end]
        next_y = self.Y[start:end]
        self.state += 1
        if self.state % self.total_batch == 0:
            self._shuffle()
        self.state %= self.total_batch
        return next_x1, next_x2, next_y
    
    def _shuffle(self):
        self.X1, self.X2, self.Y = shuffle(self.X1, self.X2, self.Y)

        
''' Define Model '''
import tensorflow as tf

# Input
context = tf.placeholder(dtype=tf.int64, shape=(None, max_seq_len), name='context')
response = tf.placeholder(dtype=tf.int64, shape=(None, max_seq_len), name='response')
target = tf.placeholder(dtype=tf.int64, shape=(None, 1), name='target')


with tf.device('/cpu:0'):
    # Embedding
    init_embedding_W = np.load(open(params['word2vec_path'], 'rb'))
    # embeddings_W = tf.Variable(init_embedding_W, name='embeddings_W')
    embedding_input_dim = init_embedding_W.shape[0]  # vocab size
    embedding_output_dim = init_embedding_W.shape[1]  # embedding output dim
    init_embedding_W = tf.constant_initializer(init_embedding_W)
    embeddings_W = tf.get_variable('embeddings_W', shape=[embedding_input_dim, embedding_output_dim], initializer=init_embedding_W)
    context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
    response_embedded = tf.nn.embedding_lookup(embeddings_W, response, name="embed_response")

with tf.device('/gpu:0'):
    # shared LSTM encoder
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=2.0, 
                use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

    c_outputs, c_states = tf.nn.dynamic_rnn(cell, context_embedded, dtype=tf.float32)
    encoding_context = c_states.h
    r_outputs, r_states = tf.nn.dynamic_rnn(cell, response_embedded, dtype=tf.float32)
    encoding_response = r_states.h

    # σ(cMr)
    M = tf.get_variable('M', shape=[params['rnn_dim'], params['rnn_dim']], initializer=tf.truncated_normal_initializer())
    # M = tf.Variable(initializer=tf.truncated_normal_initializer(), name='M')
    
    # "Predict" a  response: c * M
    generated_response = tf.matmul(encoding_context, M)
    generated_response = tf.expand_dims(generated_response, 2)
    encoding_response = tf.expand_dims(encoding_response, 2)

    # Dot product between generated response and actual response
    logits = tf.matmul(generated_response, encoding_response, True)
    logits = tf.squeeze(logits, [2])

    # Apply sigmoid to convert logits to probabilities (for prediction, not for loss)
    probs = tf.sigmoid(logits)

    # Calculate the binary cross-entropy loss
    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(target)), name='mean_loss_of_batch')

with tf.device('/cpu:0'):
    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)


''' Train '''
# Train
# from mini_batch_helper import MiniBatchCorpus
# train_data = MiniBatchCorpus()

import os
from mini_batch_helper import MiniBatch
train_data = Dataset(x1, x2, y, params['batch_size'])

exp_name = 'dual_lstm_0'
saver = tf.train.Saver()
with tf.Session() as sess:
    # Ver1:
    # sess.run(embeddings_W.initializer)
    # sess.run(M.initializer)
    # Ver2:
    # sess.run(tf.initialize_all_variables())
    # Ver3:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Restore model
    # saver.restore(sess, 'models/%s/%s.ckpt' % (exp_name, params['n_iterations'])
    # print("Model restored.")

    for it in range(params['n_iterations']):
        print('Iterations %4d:\t' %(it+1) , end="")
        # Train next batch
        next_x1, next_x2, next_y = train_data.next_batch()
        sess.run(train_step, feed_dict={context: next_x1, response: next_x2, target: next_y})

    # Save the model
    if not os.path.exists('models/'+exp_name):
        os.makedirs('models/'+exp_name)
    save_path = saver.save(sess, 'models/%s/%s.ckpt' % (exp_name, params['n_iterations']))
