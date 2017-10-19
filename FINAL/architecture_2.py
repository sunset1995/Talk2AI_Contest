
# coding: utf-8

# # Dual LSTM

# In[1]:

import tensorflow as tf
import pandas as pd
import numpy as np
import re
import jieba
jieba.set_dictionary('./dict/dict.txt.big')
jieba.load_userdict('./dict/edu_dict.txt')
import os
import time
import gc
import json
import sys


# In[2]:

input_fname = sys.argv[1]


# In[3]:

test_datas = pd.read_csv(input_fname)
test_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.dialogue.values]
test_x1 = [[[word for word in jieba.cut(s) if word.strip()] for s in q] for q in test_x1]
test_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.options.values]
test_x2 = [[[word for word in jieba.cut(s) if word.strip()] for s in rs] for rs in test_x2]

word2id = {}
with open('./dict/id2word', 'r') as f:
    for i, word in enumerate(f.readline().split(' ')):
        word2id[word] = i

def word_lst_2_id_lst(lst, pad_to_len=-1):
    pad_word_id = word2id['<pad>']
    pad_len = max(len(lst), 0)
    id_list = [word2id[lst[i]] if i<len(lst) and lst[i] in word2id else pad_word_id for i in range(pad_len)]
    pad_len = pad_to_len - len(id_list)
    if pad_len > 0:
        id_list.extend([pad_word_id] * pad_len)
    return id_list

test_id1 = [[word for s in q for word in s] for q in test_x1]
test_id1 = np.array([word_lst_2_id_lst(s) for s in test_id1])
test_id2 = [[[word for word in r] for r in rs] for rs in test_x2]
test_id2 = np.array([[word_lst_2_id_lst(r) for r in rs] for rs in test_id2])


# In[4]:

params = {}
params['embedding_shape'] = (65865, 200)
params['rnn_dim'] = 256
params['n_layers'] = 1
params['batch_size'] = 200


# In[5]:

# Define model
# Input
context = tf.placeholder(dtype=tf.int32, shape=(None, None), name='context')
context_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_len')
response = tf.placeholder(dtype=tf.int32, shape=(None, None), name='response')
response_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='response_len')
target = tf.placeholder(dtype=tf.int32, shape=(None, ), name='target')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


# Embedding
embeddings_W = tf.get_variable('embeddings_W', shape=params['embedding_shape'])
context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
response_embedded = tf.nn.embedding_lookup(embeddings_W, response, name="embed_response")

if params['n_layers'] == 1:
# shared LSTM encoder
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=1.0,
                use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    c_outputs, c_states = tf.nn.dynamic_rnn(cell, context_embedded, sequence_length=context_len, dtype=tf.float32)
    encoding_context = c_states.h
    r_outputs, r_states = tf.nn.dynamic_rnn(cell, response_embedded, sequence_length=response_len, dtype=tf.float32)
    encoding_response = r_states.h
    #mask = tf.expand_dims(tf.one_hot(response_len, depth=tf.shape(response)[1]), 1)
    #encoding_response =  tf.squeeze(tf.matmul(mask, r_outputs), 1)  # r_states.h
else:
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=1.0, use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse) 
                for _ in range(params['n_layers'])]
    dropcells = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=keep_prob) for cell in cells]
    multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=True)
    multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=keep_prob)
    c_outputs, c_states = tf.nn.dynamic_rnn(multicell, context_embedded, sequence_length=context_len, dtype=tf.float32)
    encoding_context = c_states[-1].h
    r_outputs, r_states = tf.nn.dynamic_rnn(multicell, response_embedded, sequence_length=response_len, dtype=tf.float32)
    encoding_response = r_states[-1].h

# σ(cMr)
M = tf.get_variable('M', shape=[params['rnn_dim'], params['rnn_dim']], initializer=tf.truncated_normal_initializer(stddev=0.01))

# "Predict" a  response: c * M
generated_response = tf.matmul(encoding_context, M)
generated_response = tf.expand_dims(generated_response, 2)
encoding_response = tf.expand_dims(encoding_response, 2)

# Dot product between generated response and actual response
logits = tf.matmul(generated_response, encoding_response, True)
logits = tf.reshape(logits, [-1])

# Apply sigmoid to convert logits to probabilities (for prediction, not for loss)
probs = tf.sigmoid(logits)


# In[6]:

model_names = [
    'models/dual_lstm_16/newest/model.ckpt',
    'models/dual_lstm_16/best/model.ckpt',
    'models/dual_lstm_17/newest/model.ckpt',
    'models/dual_lstm_17/best/model.ckpt',
]

def unitvec(vec):
    l = np.linalg.norm(vec)
    return vec / l if l != 0 else vec


# In[7]:

for mn in model_names:
    o_fname = '_'.join(mn.split('/')[1:3])
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, mn)
    
    # sigmoid(cMr)
    all_score = []
    for i in range(0, len(test_id1), params['batch_size']):
        q = np.repeat(test_id1[i:i+params['batch_size']], 6)
        r = test_id2[i:i+params['batch_size']].reshape(-1)
        q_l = [len(s) for s in q]
        r_l = [len(s) for s in r]
        max_l = max(q_l + r_l)
        q = np.array([[s[j] if j<len(s) else 0 for j in range(max_l)] for s in q])
        r = np.array([[s[j] if j<len(s) else 0 for j in range(max_l)] for s in r])
        now_score = sess.run(probs, {
                context: q,
                response: r,
                keep_prob: 1.0,
                context_len: q_l,
                response_len: r_l})
        all_score.extend(now_score)
    all_score = np.array(all_score).reshape(-1, 6)
    with open('__'+o_fname+'_sigmoid_cMr.txt', 'w') as fo:
        fo.write(','.join([str(a) for a in np.argmax(all_score, axis=1)]))
    
    # cossim(c, Mr)
    qq = []
    rr = []
    for i in range(0, len(test_id1), params['batch_size']):
        q = test_id1[i:i+params['batch_size']]
        r = test_id2[i:i+params['batch_size']].reshape(-1)
        q_l = [len(s) for s in q]
        r_l = [len(s) for s in r]
        max_l = max(q_l + r_l)
        q = np.array([[s[j] if j<len(s) else 0 for j in range(max_l)] for s in q])
        r = np.array([[s[j] if j<len(s) else 0 for j in range(max_l)] for s in r])
        q_state = sess.run(generated_response, {
            context: q,
            keep_prob: 1.0,
            context_len: q_l,
        })
        r_state = sess.run(encoding_response, {
            response: r,
            keep_prob: 1.0,
            response_len: r_l,
        })
        qq.extend(q_state.reshape(-1, params['rnn_dim']))
        rr.extend(r_state.reshape(-1, 6, params['rnn_dim']))
    qq = np.array(qq)
    rr = np.array(rr)

    state_cossim = []
    for q, rs in zip(qq, rr):
        for r in rs:
            state_cossim.append(np.dot(unitvec(q), unitvec(r)))
    state_cossim = np.array(state_cossim).reshape(-1, 6)
    with open('__'+o_fname+'_cossim_c_Mr.txt', 'w') as fo:
        fo.write(','.join([str(a) for a in np.argmax(state_cossim, axis=1)]))


# In[ ]:



