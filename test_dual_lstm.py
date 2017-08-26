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
from gensim.models import word2vec


exp_name = 'dual_lstm_0'  # model_to_test


# Parameters
# Model Parameters
params = {}
save_params_dir = 'models/%s/' %exp_name
params['word2vec_model_name'] = word2vec_fname
params['word2vec_vocab_size'] = embedding_matrix.shape[0]
params['word2vec_dim'] = embedding_matrix.shape[0]
params['rnn_dim'] = 512  # 256
params['n_layers'] = 1

# Training Parameters
params['learning_rate'] = 1e-4
params['keep_prob_train'] = 0.8
params['keep_prob_valid'] = 1.0
params['clip'] = 0.25
params['batch_size'] = 512  # 256  #128
params['eval_batch_size'] = 16
params['n_iterations'] = int(40 * train_data_loader.data_num / params['batch_size'])


if not os.path.exists(save_params_dir):
    os.makedirs(save_params_dir)
with open(save_params_dir+'model_parameters.json', 'w') as f:
    json.dump(params, f, indent=1)


# Define model
import tensorflow as tf

# Input
context = tf.placeholder(dtype=tf.int64, shape=(None, None), name='context')
response = tf.placeholder(dtype=tf.int64, shape=(None, None), name='response')
target = tf.placeholder(dtype=tf.int64, shape=(None, ), name='target')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

#with tf.device('/gpu:0'):
# Embedding
init_embedding_W = tf.constant_initializer(embedding_matrix)
embeddings_W = tf.get_variable('embeddings_W', shape=[embedding_matrix.shape[0], embedding_matrix.shape[1]], initializer=init_embedding_W)
context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
response_embedded = tf.nn.embedding_lookup(embeddings_W, response, name="embed_response")

if params['n_layers'] == 1:
# shared LSTM encoder
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=2.0, 
                use_peepholes=True, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    c_outputs, c_states = tf.nn.dynamic_rnn(cell, context_embedded, dtype=tf.float32)
    encoding_context = c_states.h
    r_outputs, r_states = tf.nn.dynamic_rnn(cell, response_embedded, dtype=tf.float32)
    encoding_response = r_states.h
else:
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], forget_bias=2.0, use_peepholes=True, state_is_tuple=False, reuse=tf.get_variable_scope().reuse) 
                for _ in range(params['n_layers'])]
    dropcells = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=keep_prob) for cell in cells]
    multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
    multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=keep_prob)
    c_outputs, c_states = tf.nn.dynamic_rnn(multicell, context_embedded, dtype=tf.float32)
    encoding_context = c_states.h
    r_outputs, r_states = tf.nn.dynamic_rnn(multicell, response_embedded, dtype=tf.float32)
    encoding_response = r_states.h

# σ(cMr)
M = tf.get_variable('M', shape=[params['rnn_dim'], params['rnn_dim']], initializer=tf.truncated_normal_initializer())

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
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(target)), name='mean_loss_of_batch')

#train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
optimizer = tf.train.AdamOptimizer(params['learning_rate'])
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, params['clip']), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)





# Load word2id vocab
word2vec_fname = 'models/word2vec_no_tc_offitial_200.model.bin'
extra_words = ['<pad>']
unknown_word = '<pad>'

word2vec_model = word2vec.Word2Vec.load(word2vec_fname)
# Extract word2vec
word2id = {}
id2word = [None] * (len(word2vec_model.wv.vocab) + len(extra_words)) 

for i, word in enumerate(extra_words):
    word2id[word] = i + len(word2vec_model.wv.vocab)
    id2word[i + len(word2vec_model.wv.vocab)] = word

for k, v in word2vec_model.wv.vocab.items():
    word2id[k] = v.index
    id2word[v.index] = k

del(word2vec_model)
gc.collect()



# Load in sample and test
sample = pd.read_csv('datas/sample_test_data.txt')
sample_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.dialogue.values]
sample_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.options.values]
sample_y = sample.answer.values
assert(np.sum([len(_)!=6 for _ in sample_x2]) == 0)
sample_x1 = [[word for word in jieba.cut(' '.join(s)) if word != ' '] for s in sample_x1]
sample_x2 = [[[word for word in jieba.cut(r) if word != ' '] for r in rs] for rs in sample_x2]


test_datas = pd.read_csv('datas/AIFirstProblem.txt')
test_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.dialogue.values]
test_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.options.values]
assert(np.sum([len(_)!=6 for _ in test_x2]) == 0)
test_x1 = [[word for word in jieba.cut(' '.join(s)) if word != ' '] for s in test_x1]
test_x2 = [[[word for word in jieba.cut(r) if word != ' '] for r in rs] for rs in test_x2]




def word_lst_2_id_lst(lst):
    pad_word_id = word2id['<pad>']
    return [word2id[lst[i]] if i<len(lst) and lst[i] in word2id else pad_word_id for i in range(max_seq_len)]


sample_id1 = np.array([word_lst_2_id_lst(s) for s in sample_x1])
sample_id2 = np.array([[word_lst_2_id_lst(r) for r in rs] for rs in sample_x2])
test_id1 = np.array([word_lst_2_id_lst(s) for s in test_x1])
test_id2 = np.array([[word_lst_2_id_lst(r) for r in rs] for rs in test_x2])


def eval_ans(lst_id1, lst_id2):
    score = sess.run(logits, {
        context: np.repeat(lst_id1, 6, axis=0),
        response: lst_id2.reshape(-1, max_seq_len),
        keep_prob: params['keep_prob_valid']})
    score = score.reshape(-1, 6)
    return np.argmax(score, axis=1)


def eval_on_sample():
    my_ans = eval_ans(sample_id1, sample_id2)
    return np.sum(my_ans == sample_y)


# Predict on sample dataset
saver = tf.train.Saver()
with tf.Session() as sess:
    # Restore model
    saver.restore(sess, record['best_model_dir']+'model.ckpt')
    get_valid_loss_accuracy(sess)
    sample_correct = eval_on_sample()
    my_sample_ans = eval_ans(sample_id1, sample_id2)
    my_test_ans = eval_ans(test_id1, test_id2)
    print('sample correct %4d' % (sample_correct), flush=True)
    record['sample_correct'] = sample_correct.tolist()

