
# coding: utf-8

# In[2]:

import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from scipy import spatial
from scipy import stats

# Import & Init jieba
import jieba
jieba.set_dictionary('./dict/dict.txt.big')
jieba.load_userdict('./dict/edu_dict.txt')

# Import pandas
import pandas as pd

# Import util
import time
import re
import sys


# In[7]:

input_fname = sys.argv[1]
output_fname = sys.argv[2]


# In[8]:

test_datas = pd.read_csv(input_fname)
test_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.dialogue.values]
test_x1 = [[[word for word in jieba.cut(s) if word.strip()] for s in q] for q in test_x1]
test_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.options.values]
test_x2 = [[[word for word in jieba.cut(s) if word.strip()] for s in rs] for rs in test_x2]


# In[3]:

model_names = [
    'word2vec/dual-lstm-13-newest',
    'word2vec/smn-1-best',
]


# In[4]:

# Naive - centroid
def unitvec(vec):
    l = np.linalg.norm(vec)
    return vec / l if l != 0 else vec

def centroid(sentence):
    vecs = [word_vectors.word_vec(word) for word in sentence if word in word_vectors.vocab]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(word_vectors.vector_size)

def centroid_score(x1, x2):
    cos_score = []
    for a, b in zip(x1, x2):
        a_sentence = [word for s in a for word in s]
        b_sentences = [[word for word in s] for s in b]

        a_center = centroid(a_sentence)
        b_centers = [centroid(s) for s in b_sentences]

        cos_score.append([np.dot(unitvec(a_center), unitvec(bc)) for bc in b_centers])
    return np.array(cos_score).reshape(-1, 6)

def attack_naive_centroid(x1, x2):
    my_cos_ans = centroid_score(x1, x2)
    return np.argmax(my_cos_ans, axis=1)


# In[5]:

# Naive - sentence decay centroid
def dis_centroid(ss, beta=0.77):
    for s in ss:
        assert(type(s) == list)
    vecs = [[word_vectors.word_vec(word) for word in s if word in word_vectors.vocab] for s in ss]
    vecs = [s for s in vecs if len(s) > 0]
    if len(vecs) == 0:
        return np.zeros(word_vectors.vector_size)
    cens = list(reversed([np.mean(vs, axis=0) for vs in vecs]))
    for cen in cens:
        assert(np.sum(np.isnan(cen)) == 0)
    w_sum = sum(beta**i for i in range(len(cens)))
    return np.sum([cens[i] * (beta ** i / w_sum) for i in range(len(cens))], axis=0)

def dis_centroid_score(x1, x2):
    cos_score = []
    for a, b in zip(x1, x2):
        a_sentence = [[word for word in s] for s in a]
        b_sentences = [[word for word in s] for s in b]

        a_center = dis_centroid(a_sentence)
        b_centers = [dis_centroid([s]) for s in b_sentences]

        cos_score.append([np.dot(unitvec(a_center), unitvec(bc)) for bc in b_centers])
    return np.array(cos_score).reshape(-1, 6)

def attack_naive_dis_centroid(x1, x2):
    my_cos_ans = dis_centroid_score(x1, x2)
    return np.argmax(my_cos_ans, axis=1)


# In[6]:

# Naive - word decay centroid
def w_centroid(ss, beta=0.77):
    for s in ss:
        assert(type(s) == list)
    vecs = [[word_vectors.word_vec(word) for word in s if word in word_vectors.vocab] for s in ss]
    vecs = list(reversed([s for s in vecs if len(s) > 0]))
    w_cen = np.zeros(word_vectors.vector_size)
    if len(vecs) == 0:
        return w_cen
    w = np.array([beta**i for i in range(len(vecs)) for _ in range(len(vecs[i]))]).reshape(-1, 1)
    cen = np.array([vec for s in vecs for vec in s])
    return np.sum(w * cen, axis=0) / np.sum(w)

def w_centroid_score(x1, x2):
    cos_score = []
    for a, b in zip(x1, x2):
        a_sentence = [[word for word in s] for s in a]
        b_sentences = [[word for word in s] for s in b]

        a_center = w_centroid(a_sentence)
        b_centers = [w_centroid([s]) for s in b_sentences]

        cos_score.append([np.dot(unitvec(a_center), unitvec(bc)) for bc in b_centers])
    return np.array(cos_score).reshape(-1, 6)

def attack_naive_w_centroid(x1, x2):
    my_cos_ans = w_centroid_score(x1, x2)
    return np.argmax(my_cos_ans, axis=1)


# In[10]:

# Loading model
word_vec_models = [KeyedVectors.load(mn) for mn in model_names]
word2id = {}
for w, idx in word_vec_models[0].vocab.items():
    word2id[w] = idx.index

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



# Define tf model
params = {}
params['embedding_shape'] = (65865, 200)
params['rnn_dim'] = 256
params['n_layers'] = 2
params['forget_bias'] = 1.0

# Input
context = tf.placeholder(dtype=tf.int32, shape=(None, None), name='context')
context_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_len')
response = tf.placeholder(dtype=tf.int32, shape=(None, None), name='response')
response_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='response_len')
target = tf.placeholder(dtype=tf.int32, shape=(None, ), name='target')

# Embedding
embeddings_W = tf.get_variable('embeddings_W', shape=params['embedding_shape'])
context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
response_embedded = tf.nn.embedding_lookup(embeddings_W, response, name="embed_response")

if params['n_layers'] == 1:
    # shared LSTM encoder
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=params['forget_bias'], 
                use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0)
    c_outputs, c_states = tf.nn.dynamic_rnn(cell, context_embedded, sequence_length=context_len, dtype=tf.float32)
    encoding_context = c_states.h
    r_outputs, r_states = tf.nn.dynamic_rnn(cell, response_embedded, sequence_length=response_len, dtype=tf.float32)
    encoding_response = r_states.h
    #mask = tf.expand_dims(tf.one_hot(response_len, depth=tf.shape(response)[1]), 1)
    #encoding_response =  tf.squeeze(tf.matmul(mask, r_outputs), 1)  # r_states.h
else:
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=params['forget_bias'], use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse) 
                for _ in range(params['n_layers'])]
    dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0) for cell in cells]
    multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=True)
    multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=1.0)
    c_outputs, c_states = tf.nn.dynamic_rnn(multicell, context_embedded, sequence_length=context_len, dtype=tf.float32)
    encoding_context = c_states[-1].h
    r_outputs, r_states = tf.nn.dynamic_rnn(multicell, response_embedded, sequence_length=response_len, dtype=tf.float32)
    encoding_response = r_states[-1].h
    
# σ(cMr)
M = tf.get_variable('M', shape=[params['rnn_dim'], params['rnn_dim']])

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


saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'dual_model/model.ckpt')




# Cocu ans
mode_ans = []
for i, (q, rs) in enumerate(zip(test_x1, test_x2)):
    q_id = test_id1[i]
    rs_id = test_id2[i]
    now_ans = []
    
    # Naive
    word_vectors = word_vec_models[0]
    now_ans.extend([
        attack_naive_dis_centroid([q], [rs])[0],
        attack_naive_w_centroid([q], [rs])[0],
    ])
    word_vectors = word_vec_models[1]
    now_ans.extend([
        attack_naive_dis_centroid([q], [rs])[0],
        attack_naive_w_centroid([q], [rs])[0],
    ])
    
    # sigmoid(cMr)
    cMr_score = []
    for r_id in rs_id:
        now_score = sess.run(probs, {
            context: [q_id],
            response: [r_id],
            context_len:[len(q_id)],
            response_len:[len(r_id)]})[0]
        cMr_score.append(now_score)
    now_ans.append(np.argmax(cMr_score))
    
    # cossim(c, Mr)
    cossim = []
    q_state = sess.run(generated_response, {
        context: [q_id],
        context_len: [len(q_id)]
    })[0]
    q_state = q_state.reshape(-1)
    for r in rs:
        r_state = sess.run(encoding_response, {
            response: [r_id],
            response_len: [len(r_id)]
        })[0]
        r_state = r_state.reshape(-1)
        cossim.append(np.dot(unitvec(q_state), unitvec(r_state)))
    now_ans.append(np.argmax(cossim))
    
    mode_ans.append(stats.mode(now_ans).mode[0])
    
    
    
    
    
    
# In[14]:

with open(output_fname, 'w') as f:
    f.write('id,ans\n')
    f.write('\n'.join(['%d,%d' % (i+1, a) for i, a in enumerate(mode_ans)]))
    f.write('\n')



