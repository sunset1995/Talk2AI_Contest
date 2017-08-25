import numpy as np
import tensorflow as tf
from scipy import spatial
from scipy import stats
from gensim.models import word2vec

# Import pandas
import pandas as pd
from pandas import Series, DataFrame

# Import util
import time
import re
import sys
import gc

# Self define module
from mini_batch_helper import extractor
from mini_batch_helper import MiniBatchCorpus



word2vec_fname = 'models/word2vec_no_tc_offitial_200.model.bin'
corpus_fnames = [
    'datas/training_data/no_TC_下課花路米.txt',
    'datas/training_data/no_TC_人生劇展.txt',
    'datas/training_data/no_TC_公視藝文大道.txt',
    'datas/training_data/no_TC_成語賽恩思.txt',
    'datas/training_data/no_TC_我的這一班.txt',
    'datas/training_data/no_TC_流言追追追.txt',
    'datas/training_data/no_TC_聽聽看.txt',
    'datas/training_data/no_TC_誰來晚餐.txt',
]
extra_words = ['<pad>']
unknown_word = '<pad>'

word2id, id2word, word_p, embedding_matrix, corpus, corpus_id = extractor(word2vec_fname, corpus_fnames, extra_words, unknown_word)

voc_size = embedding_matrix.shape[0]
emb_size = embedding_matrix.shape[1]
unknown_word_id = word2id['<pad>']
pad_word_id = word2id['<pad>']
max_seq_len = np.max([len(s) for cp in corpus_id for s in cp])

print('%20s: %d' % ('unknown_word_id', unknown_word_id))
print('%20s: %d' % ('pad_word_id', pad_word_id))
print('%20s: %d' % ('max_seq_len', max_seq_len))

# Data split
rnd_idx = np.arange(len(corpus_id))
np.random.shuffle(rnd_idx)
corpus_id = corpus_id[rnd_idx[:len(corpus_id)//2]]
valid_corpus_num = 10

train_data_loader = MiniBatchCorpus(corpus_id[valid_corpus_num:])
valid_data_loader = MiniBatchCorpus(corpus_id[:valid_corpus_num])
print('train datas num:', train_data_loader.data_num)
print('valid datas num:', valid_data_loader.data_num)



# Word embedding model
tf_word_p = tf.constant(word_p, dtype=tf.float64)
embeddings_W = tf.Variable(embedding_matrix)
del(embedding_matrix)
gc.collect()

# Input
wa = tf.placeholder(tf.float64, [1])
x1 = tf.placeholder(tf.int32, [None, max_seq_len])
x2 = tf.placeholder(tf.int32, [None, max_seq_len])
y = tf.placeholder(tf.float64, [None])

def sentence_embedding(xs):
    xs_mask = 1 - tf.to_double(tf.equal(xs, pad_word_id))
    xs_len = tf.reduce_sum(xs_mask, axis=1)
    xs_embedded = tf.gather(embeddings_W, xs) * tf.reshape(xs_mask, [-1, max_seq_len, 1])
    xs_word_p = tf.gather(tf_word_p, xs)
    xs_weighted = tf.reshape(wa / (wa + xs_word_p), [-1, max_seq_len, 1]) * xs_embedded
    xs_center = tf.reduce_sum(xs_weighted, axis=1) / tf.reshape(tf.to_double(xs_len)+1e-6, [-1, 1])
    return xs_center

x1_center = sentence_embedding(x1)
x2_center = sentence_embedding(x2)
W = tf.Variable(tf.truncated_normal([emb_size, emb_size], stddev=1e-6, dtype=tf.float64))
tf_score = tf.reduce_sum((x2_center * (x1_center @ W)), axis=1)

reg = tf.nn.l2_loss(W) * 7.7 / (emb_size * emb_size)
cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf_score)) + reg
optimizer = tf.train.AdamOptimizer(1e-3)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_norm(grad, 0.2), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def eval_valid_loss():
    valid_loss = 0
    valid_batch = 1024
    batch_num = valid_data_loader.data_num // valid_batch
    for i in range(batch_num):
        b_x1, b_x2, b_y = valid_data_loader.next_batch(valid_batch, max_seq_len, pad_word_id)
        now_loss = sess.run(cost, {wa: [1e-4], x1: b_x1, x2: b_x2, y: b_y})
        valid_loss += now_loss / (batch_num * valid_batch)
    return valid_loss

batch_size = 256
epoch_num = 2
log_interval = 200
save_interval = 1000

train_batch_loss = 0
start_time = time.time()
for i_batch in range(epoch_num * train_data_loader.data_num // batch_size):
    b_x1, b_x2, b_y = train_data_loader.next_batch(batch_size, max_seq_len, pad_word_id)
    _, now_loss = sess.run([train_step, cost], {wa: [1e-4], x1: b_x1, x2: b_x2, y: b_y})
    train_batch_loss += now_loss / (log_interval * batch_size)
    if (i_batch+1) % log_interval == 0:
        valid_loss = eval_valid_loss()
        print('train batch loss %10f / valid loss %10f / elapsed time %.f' % (
            train_batch_loss, valid_loss, time.time()-start_time), flush=True)
        train_batch_loss = 0
    if save_interval is not None and (i_batch+1) % save_interval == 0:
        saver.save(sess, 'models/Attack-sentence-embedding/s_emb', global_step=i_batch+1)
        print('model saved (latest)', flush=True)

saver.save(sess, 'models/Attack-sentence-embedding/s_emb_final')