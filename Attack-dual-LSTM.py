
# coding: utf-8

# # Dual LSTM

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import re
import jieba
jieba.set_dictionary('datas/dict/dict.txt.big')
jieba.load_userdict('datas/dict/edu_dict.txt')
import os
import time
import gc
import json
from mini_batch_helper import extractor, MiniBatchCorpus


# ## Read in  training data

# In[2]:


# Read in  training data
word2vec_fname = 'models/word2vec/fine-tuned-2.txt'
corpus_fnames = [
    'datas/training_data/下課花路米.txt',
    'datas/training_data/人生劇展.txt',
    'datas/training_data/公視藝文大道.txt',
    'datas/training_data/成語賽恩思.txt',
    'datas/training_data/我的這一班.txt',
    'datas/training_data/流言追追追.txt',
    'datas/training_data/聽聽看.txt',
    'datas/training_data/誰來晚餐.txt',
]
sample_rate_on_training_datas = 1.0  # 1.0
extra_words = ['<pad>']
unknown_word = None

word2id, id2word, word_p, embedding_matrix, corpus, corpus_id = extractor(word2vec_fname, corpus_fnames, sample_rate_on_training_datas, extra_words, unknown_word)


# In[3]:


# Data split
rnd_idx = np.arange(len(corpus_id))
np.random.shuffle(rnd_idx)
corpus_id = corpus_id[rnd_idx[:len(corpus_id)]]
valid_corpus_num = 10

train_data_loader = MiniBatchCorpus(corpus_id[valid_corpus_num:], x1_len=3)
valid_data_loader = MiniBatchCorpus(corpus_id[:valid_corpus_num], x1_len=3)
print('train datas num:', train_data_loader.data_num, flush=True)
print('valid datas num:', valid_data_loader.data_num, flush=True)


# In[4]:


from pandas import Series


# In[5]:


Series([len(sentence) for episode in corpus_id for sentence in episode]).describe()


# In[6]:


max_seq_len = max([len(sentence) for episode in corpus_id for sentence in episode])
max_seq_len

del(corpus)
gc.collect()


# In[7]:


type(max_seq_len)


# ## Model ( tf )

# In[8]:


# reference: https://github.com/dennybritz/chatbot-retrieval/blob/8b1be4c2e63631b1180b97ef927dc2c1f7fe9bea/udc_hparams.py
exp_name = 'dual_lstm_19'
# Model Parameters
params = {}
save_params_dir = 'models/%s/' %exp_name
params['word2vec_model_name'] = word2vec_fname
params['word2vec_vocab_size'] = embedding_matrix.shape[0]
params['word2vec_dim'] = embedding_matrix.shape[1]
params['rnn_dim'] = 256  # 256, 384, 512
params['n_layers'] = 2
params['forget_bias'] = 1.0

# Training Parameters
params['learning_rate'] = 1e-4
params['keep_prob_train'] = 0.8 # 0.8
params['keep_prob_valid'] = 1.0
params['l1_loss'] = 1e-4 #1e-6 # regularize M
params['clip'] = 15  # 1e-2
params['batch_size'] = 256 #512
params['eval_batch_size'] = 16
params['n_iterations'] = int(20 * train_data_loader.data_num / params['batch_size'])


# In[9]:


if not os.path.exists(save_params_dir):
    os.makedirs(save_params_dir)
with open(save_params_dir+'model_parameters.json', 'w') as f:
    json.dump(params, f, indent=1)


# In[10]:


record = {}
save_record_dir = 'models/%s/' %exp_name
record['newest_model_dir'] = 'models/' + exp_name +'/newest/'
record['best_model_dir'] = 'models/' + exp_name +'/best/'
record['loss_train'] = []
record['loss_valid'] = []
record['accuracy_valid'] = []
record['best_iter'] = 0
record['sample_correct'] = 0


# ## TODO: Embedding 後可以考慮加一層 Dropout

# In[11]:


# Define model
import tensorflow as tf

# Input
context = tf.placeholder(dtype=tf.int32, shape=(None, None), name='context')
context_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_len')
response = tf.placeholder(dtype=tf.int32, shape=(None, None), name='response')
response_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='response_len')
target = tf.placeholder(dtype=tf.int32, shape=(None, ), name='target')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


#with tf.device('/gpu:0'):
# Embedding
init_embedding_W = tf.constant_initializer(embedding_matrix)
embeddings_W = tf.get_variable('embeddings_W', shape=[embedding_matrix.shape[0], embedding_matrix.shape[1]], initializer=init_embedding_W)
context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
response_embedded = tf.nn.embedding_lookup(embeddings_W, response, name="embed_response")

if params['n_layers'] == 1:
# shared LSTM encoder
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=params['forget_bias'], 
                use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    c_outputs, c_states = tf.nn.dynamic_rnn(cell, context_embedded, sequence_length=context_len, dtype=tf.float32)
    encoding_context = c_states.h
    r_outputs, r_states = tf.nn.dynamic_rnn(cell, response_embedded, sequence_length=response_len, dtype=tf.float32)
    encoding_response = r_states.h
    #mask = tf.expand_dims(tf.one_hot(response_len, depth=tf.shape(response)[1]), 1)
    #encoding_response =  tf.squeeze(tf.matmul(mask, r_outputs), 1)  # r_states.h
else:
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=params['forget_bias'], use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse) 
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
correct_prediction = tf.logical_or( tf.logical_and(tf.equal(target,1), tf.greater_equal(probs,0.5)), tf.logical_and(tf.equal(target,0), tf.less(probs,0.5)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Calculate the binary cross-entropy loss
target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(target)))
l1_loss = params['l1_loss'] * tf.reduce_sum(tf.abs(M))
loss = target_loss + l1_loss

#train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
optimizer = tf.train.Adam(params['learning_rate'])
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, params['clip']), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)


# In[12]:


def get_valid_loss_accuracy(sess):
    valid_loss = 0
    valid_accuracy = 0
    n_iter = int(valid_data_loader.data_num/params['batch_size'])
    for iter in range(n_iter):
        next_x1, next_x2, next_y, x1_len, x2_len = valid_data_loader.next_batch(
            batch_size=params['batch_size'], pad_to_length=max_seq_len, pad_word=word2id['<pad>'], return_len=True)
        new_accuracy, new_loss = sess.run([accuracy, loss], 
                                    feed_dict={context: next_x1, response: next_x2, target: next_y, 
                                    keep_prob: params['keep_prob_train'], context_len: x1_len, response_len:x2_len}) 
        valid_accuracy += new_accuracy
        valid_loss += new_loss
    valid_loss /= n_iter
    valid_accuracy /= n_iter
    print('Valid loss = %.5f, accuracy = %.5f' % (valid_loss, valid_accuracy), flush=True)
    record['loss_valid'].append( valid_loss.tolist() )
    record['accuracy_valid'].append( valid_accuracy.tolist() )
    return valid_loss


# In[13]:


# Train
start_time = time.time()
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Restore model
    # saver.restore(sess, record['best_model_dir']+'model.ckpt')
    # print('Retrain model: %s' %record['best_model_dir'], flush=True)
    best_valid_loss = 1e9
    for it in range(params['n_iterations']):
        print('Iterations %4d:\t' %(it+1) , end='', flush=True)
        # Train next batch
        next_x1, next_x2, next_y, x1_len, x2_len = train_data_loader.next_batch(batch_size=params['batch_size'], pad_to_length=max_seq_len, pad_word=word2id['<pad>'], return_len=True)
        batch_loss, batch_l1_loss, _ = sess.run([target_loss, l1_loss, train_step], 
                            feed_dict={context: next_x1, response: next_x2, target: next_y, 
                            keep_prob: params['keep_prob_train'], context_len: x1_len, response_len:x2_len}) 
        print('loss = %.5f / l1_loss = %.5f / elapsed time %.f' % (batch_loss, batch_l1_loss, time.time() - start_time), flush=True)
        record['loss_train'].append( batch_loss.tolist() )
        if it % 1000 == 0:
            # Save the model if has smaller loss
            current_valid_loss = get_valid_loss_accuracy(sess)
            if current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                if not os.path.exists(record['best_model_dir']):
                    os.makedirs(record['best_model_dir'])
                save_path = saver.save(sess, record['best_model_dir']+'model.ckpt')
                record['best_iter'] = it
                print('Best model save in %d iteration' %it, flush=True)
        if it % 100 == 0:
            if not os.path.exists(record['newest_model_dir']):
                os.makedirs(record['newest_model_dir'])
            save_path = saver.save(sess, record['newest_model_dir']+'model.ckpt')


# In[ ]:


# Dump record file as .json
if not os.path.exists(save_record_dir):
    os.makedirs(save_record_dir)
with open(save_record_dir+'%d.json' %params['n_iterations'], 'w') as f:
    json.dump(record, f, indent=1)


# In[ ]:




