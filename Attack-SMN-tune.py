
# coding: utf-8

# In[1]:


from gensim.models import word2vec
import numpy as np
from scipy import spatial

# Import & Init jieba
import jieba
jieba.set_dictionary('datas/dict/dict.txt.big')
jieba.load_userdict('datas/dict/edu_dict.txt')

# Import pandas
import pandas as pd
from pandas import Series, DataFrame

# Import util
import os
import re
import json
import time

from mini_batch_helper import extractor, MiniBatchCorpus


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


valid_corpus_num = 10

train_data_loader = MiniBatchCorpus(corpus_id[valid_corpus_num:], context_len=3, max_len=64)
valid_data_loader = MiniBatchCorpus(corpus_id[:valid_corpus_num], context_len=3, max_len=64)
print('train datas num:', train_data_loader.data_num, flush=True)
print('valid datas num:', valid_data_loader.data_num, flush=True)


# In[4]:


exp_name = 'SMN_1'
# HyperParameters
# Model Parameters
hp = {}

hp['word2vec_model_name'] = word2vec_fname
hp['word2vec_vocab_size'] = embedding_matrix.shape[0]
hp['word2vec_dim'] = embedding_matrix.shape[1]
hp['rnn_dim'] = 256  # 200
hp['forget_bias'] = 1.0 # 0.0

hp['word_len'] = 64
hp['filter_size'] = 3
hp['stride_size'] = 1
hp['fm1_num'] = 4  
hp['fm2_num'] = 8
hp['cell_type'] = 'gru'  # 'gru' or 'lstm'
hp['keep_prob'] = 0.8  # 0.8 , 0.5 !
# hp['fm1_size'] = int(hp['word_len']/(2*hp['stride_size']))  # unused ?? 
# hp['fm2_size'] = int(hp['word_len']/(2*hp['stride_size'])/(2*hp['stride_size']))

# Training Parameters
hp['learning_rate'] = 1e-3
hp['decay_learning_rate'] = 0.8
hp['decay_times_no_improve'] = 5
hp['clip'] = 15
hp['batch_size'] = 256
hp['n_iterations'] = int(20 * train_data_loader.data_num / hp['batch_size'])


# In[5]:


# Export the hyperparameters as json
save_hp_dir = 'models/%s/' %exp_name
newest_model_dir = save_hp_dir + 'newest/'
best_model_dir = save_hp_dir + 'best/'
if not os.path.exists(save_hp_dir):
    os.makedirs(save_hp_dir)
with open(save_hp_dir+'model_parameters.json', 'w') as f:
    json.dump(hp, f, indent=1)


# In[6]:


# Load in sample
sample = pd.read_csv('datas/sample_test_data.txt')
sample_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.dialogue.values]
sample_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.options.values]
sample_y = sample.answer.values
assert(np.sum([len(_)!=6 for _ in sample_x2]) == 0)
sample_len1 = np.array([len(lst) for lst in sample_x1])
sample_len2 = np.array([[len(lst) for lst in opt] for opt in sample_x2])
sample_x1 = [[word for word in jieba.cut(' '.join(s)) if word != ' '] for s in sample_x1]
sample_x2 = [[[word for word in jieba.cut(r) if word != ' '] for r in rs] for rs in sample_x2]

test_datas = pd.read_csv('datas/AIFirstProblem.txt')
test_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.dialogue.values]
test_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.options.values]
assert(np.sum([len(_)!=6 for _ in test_x2]) == 0)
test_len1 = np.array([len(lst) for lst in test_x1])
test_len2 = np.array([[len(lst) for lst in opt] for opt in test_x2])
test_x1 = [[word for word in jieba.cut(' '.join(s)) if word != ' '] for s in test_x1]
test_x2 = [[[word for word in jieba.cut(r) if word != ' '] for r in rs] for rs in test_x2]
with open('datas/AIFirst_test_answer.txt', 'r') as f:
    f.readline()
    test_y = np.array([int(line.strip().split(',')[-1]) for line in f])

def word_lst_2_id_lst(lst, pad_to_len=-1):
    pad_word_id = word2id['<pad>']
    pad_len = max(len(lst), 0)
    id_list = [word2id[lst[i]] if i<len(lst) and lst[i] in word2id else pad_word_id for i in range(pad_len)]
    pad_len = pad_to_len - len(id_list)
    if pad_len > 0:
        id_list.extend([pad_word_id] * pad_len)
    return id_list

pad_to_length = hp['word_len']

sample_id1 = np.array([word_lst_2_id_lst(s, pad_to_length) for s in sample_x1])
sample_id2 = np.array([[word_lst_2_id_lst(r, pad_to_length) for r in rs] for rs in sample_x2])
test_id1 = np.array([word_lst_2_id_lst(s, pad_to_length) for s in test_x1])
test_id2 = np.array([[word_lst_2_id_lst(r, pad_to_length) for r in rs] for rs in test_x2])

# In[7]:


# Define model
import tensorflow as tf

def compute_accuracy(next_x1, next_x2, _y, _keep_prob):
    global prediction
    y_pre = sess.run(prediction, feed_dict={context: next_x1, response: next_x2, keep_prob:_keep_prob})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={context: next_x1, response: next_x2, target: _y, keep_prob:_keep_prob})
    return result
 
def weight_variable(shape):
    initial = tf.random_uniform(shape,-1.0,1.0)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.random_uniform(shape,-1.0,1.0)
    return tf.Variable(initial)
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, hp['stride_size'], hp['stride_size'], 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Input
context = tf.placeholder(dtype=tf.int32, shape=(None, None), name='context')
context_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_len')
response = tf.placeholder(dtype=tf.int32, shape=(None, None), name='response')
response_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='response_len')
target = tf.placeholder(dtype=tf.int32, shape=(None,), name='target')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')


# In[8]:


#embedding
init_embedding_W = tf.constant_initializer(embedding_matrix)
embeddings_W = tf.get_variable('embeddings_W', shape=[embedding_matrix.shape[0], embedding_matrix.shape[1]], initializer=init_embedding_W)
context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
response_embedded = tf.nn.embedding_lookup(embeddings_W, response, name="embed_response")
# here should pass a gru


# In[9]:


# rnn layer
assert(hp['cell_type'] == 'gru' or hp['cell_type'] == 'lstm')
if hp['cell_type'] == 'gru':
    cell = tf.contrib.rnn.GRUCell(num_units=hp['rnn_dim'], reuse=tf.get_variable_scope().reuse)
elif hp['cell_type'] == 'lstm':
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hp['rnn_dim'], forget_bias=hp['forget_bias'], 
                                   use_peepholes=True, state_is_tuple=True, 
                                   reuse=tf.get_variable_scope().reuse)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
c_outputs, c_states = tf.nn.dynamic_rnn(cell, context_embedded, sequence_length=context_len, dtype=tf.float32)
context_rnn = c_outputs
r_outputs, r_states = tf.nn.dynamic_rnn(cell, response_embedded, sequence_length=response_len, dtype=tf.float32)
response_rnn = r_outputs


# In[10]:


# M1 matrix and M2 matrix

# M1 word dot matrix
word_dot_matrix = tf.matmul(context_embedded, response_embedded, False, True)
m1_image = tf.reshape(word_dot_matrix, [-1, hp['word_len'], hp['word_len'], 1])
m1_image = tf.divide(m1_image, 1e-9 + tf.reshape(tf.reduce_max(m1_image, axis=[1, 2]), [-1, 1, 1, 1]))

# M2 segment dot matrix
segment_dot_matrix = tf.matmul(context_rnn, response_rnn, False, True)
m2_image = tf.reshape(segment_dot_matrix, [-1, hp['word_len'], hp['word_len'], 1])
m2_image = tf.divide(m2_image, 1e-9 + tf.reshape(tf.reduce_max(m2_image, axis=[1, 2]), [-1, 1, 1, 1]))

y_label=tf.cast(target, tf.float32)
# M1 convolution
W_conv1_m1 = weight_variable([hp['filter_size'], hp['filter_size'], 1, hp['fm1_num']])
b_conv1_m1 = bias_variable([hp['fm1_num']])
h_conv1_m1 = tf.nn.relu(conv2d(m1_image, W_conv1_m1) + b_conv1_m1)
h_pool1_m1 = max_pool_2x2(h_conv1_m1)

W_conv2_m1 = weight_variable([hp['filter_size'], hp['filter_size'], hp['fm1_num'], hp['fm2_num']])
b_conv2_m1 = bias_variable([hp['fm2_num']])
h_conv2_m1 = tf.nn.relu(conv2d(h_pool1_m1, W_conv2_m1) + b_conv2_m1)
h_pool2_m1 = max_pool_2x2(h_conv2_m1)

h_pool2_m1_flat = tf.contrib.layers.flatten(h_pool2_m1)
# tf.reshape(, [-1, hp['fm2_size']*hp['fm2_size']*hp['fm2_num']])  # ??

# M2 convolution
W_conv1_m2 = weight_variable([hp['filter_size'], hp['filter_size'], 1, hp['fm1_num']])
b_conv1_m2 = bias_variable([hp['fm1_num']])
h_conv1_m2 = tf.nn.relu(conv2d(m2_image, W_conv1_m2) + b_conv1_m2)
h_pool1_m2 = max_pool_2x2(h_conv1_m2)

W_conv2_m2 = weight_variable([hp['filter_size'], hp['filter_size'], hp['fm1_num'], hp['fm2_num']])
b_conv2_m2 = bias_variable([hp['fm2_num']])
h_conv2_m2 = tf.nn.relu(conv2d(h_pool1_m2, W_conv2_m2) + b_conv2_m2)
h_pool2_m2 = max_pool_2x2(h_conv2_m2)

h_pool2_m2_flat = tf.contrib.layers.flatten(h_pool2_m2)
# tf.reshape(h_pool2_m2, [-1, hp['fm2_size']*hp['fm2_size']*hp['fm2_num']])

# Accumulate M1 and M2
matching_accumulation = tf.add(h_pool2_m1_flat, h_pool2_m2_flat)

W_fc1 = weight_variable([int(matching_accumulation.shape[1]), hp['word_len']*hp['word_len']])
b_fc1 = bias_variable([hp['word_len']*hp['word_len']])
h_fc1 = tf.nn.sigmoid(tf.matmul(matching_accumulation, W_fc1) + b_fc1)

W_fc2 = weight_variable([hp['word_len']*hp['word_len'], 1])
b_fc2 = bias_variable([1])
logits = tf.reshape(tf.matmul(h_fc1, W_fc2) + b_fc2, [-1])


# In[11]:


# Apply sigmoid to convert logits to probabilities (for prediction, not for loss)
probs = tf.sigmoid(logits)
correct_prediction = tf.logical_or( tf.logical_and(tf.equal(target,1), tf.greater_equal(probs,0.5)), tf.logical_and(tf.equal(target,0), tf.less(probs,0.5)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimize
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_label)
optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(cross_entropy)
capped_gvs = [(tf.clip_by_value(grad, -hp['clip'], hp['clip']), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)


# In[ ]:


def problem_acc(sess, q, q_len, rs, rs_len, ans):
    p_prob = sess.run(probs, feed_dict={
            context: np.repeat(q, 6, axis=0).reshape(-1, hp['word_len']),
            context_len: np.repeat(q_len, 6), 
            response: rs.reshape(-1, hp['word_len']),
            response_len: rs_len.reshape(-1),
            keep_prob: 1.0,})
    return np.sum(np.argmax(p_prob.reshape(-1, 6), axis=1) == ans) / len(ans)


# In[58]:


def get_valid_loss_accuracy(sess):
    valid_loss = 0
    valid_accuracy = 0
    n_iter = int(valid_data_loader.data_num/hp['batch_size'])
    for it in range(n_iter):
        next_x1, next_x2, next_y, x1_len, x2_len = train_data_loader.next_batch(
            batch_size=hp['batch_size'], pad_to_length=hp['word_len'], pad_word=word2id['<pad>'], return_len=True)
        batch_loss, batch_acc = sess.run([cross_entropy, accuracy], feed_dict={
            context: next_x1, response: next_x2, target: next_y,
            keep_prob: hp['keep_prob'], context_len: x1_len, response_len:x2_len, learning_rate:lr})
        batch_loss = np.mean(batch_loss)
        valid_accuracy += batch_acc
        valid_loss += batch_loss
    valid_loss /= n_iter
    valid_accuracy /= n_iter
    print('Valid loss = %.5f, accuracy = %.5f' % (valid_loss, valid_accuracy), flush=True)
    print('Sample accuracy = %.5f' % problem_acc(sess, sample_id1, sample_len1, sample_id2, sample_len2, sample_y))
    return valid_loss


# In[13]:


saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[16]:


start_time = time.time()
lr = hp['learning_rate']
decay_times_no_improve = hp['decay_times_no_improve']
best_valid_loss = 1e9
for it in range(hp['n_iterations']):
    print('Iterations %4d:\t' % (it+1) , end='')
    next_x1, next_x2, next_y, x1_len, x2_len = train_data_loader.next_batch(
        batch_size=hp['batch_size'], pad_to_length=hp['word_len'], pad_word=word2id['<pad>'], return_len=True)
    batch_loss, batch_acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict={
        context: next_x1, response: next_x2, target: next_y,
        keep_prob: hp['keep_prob'], context_len: x1_len, response_len: x2_len, learning_rate: lr})
    batch_loss = np.mean(batch_loss)
    print('Train loss = %.5f, accuracy = %.5f / elapsed time %.f' % (batch_loss, batch_acc, time.time() - start_time), flush=True)
    if it % 1000 == 0:
        # Save the model if has smaller loss
        current_valid_loss = get_valid_loss_accuracy(sess)
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            save_path = saver.save(sess, best_model_dir+'model.ckpt')
            print('Best model save in %d iteration' % (it+1), flush=True)

        # Decay the learning rate if no improve for 3 times
        if hp['decay_learning_rate'] < 1:
            if current_valid_loss > best_valid_loss:
                times_no_improve += 1
            else:
                times_no_improve = 0
                decay_times_no_improve = max(hp['decay_times_no_improve'], decay_times_no_improve-1)
            if times_no_improve >= decay_times_no_improve:
                # Decay learning rate
                times_no_improve = 0
                decay_times_no_improve *= 2
                lr *= hp['decay_learning_rate']
                print('Learning rate decay to %f' % lr, flush=True)
                # Restrore to the best model
                saver.restore(sess, best_model_dir+'model.ckpt')
                # Stop if lr is too small
                if lr < 1e-9:
                    print('Current_learning_rate is smaller than 1e-9. Stop.', flush=True)
                    break
    if it % 100 == 0:
        if not os.path.exists(newest_model_dir):
            os.makedirs(newest_model_dir)
        save_path = saver.save(sess, newest_model_dir+'model.ckpt')


# In[ ]:




