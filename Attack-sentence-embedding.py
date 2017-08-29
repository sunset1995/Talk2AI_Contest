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



word2vec_fname = 'models/word2vec/vec200_win40_iter15_mincnt5.bin'
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
sample_rate_on_training_datas = 1
extra_words = ['<pad>']

word2id, id2word, word_p, embedding_matrix, corpus, corpus_id = extractor(word2vec_fname, corpus_fnames, sample_rate_on_training_datas, extra_words, unknown_word=None)



voc_size = embedding_matrix.shape[0]
emb_size = embedding_matrix.shape[1]
pad_word_id = word2id['<pad>']
max_seq_len = np.max([len(s) for cp in corpus_id for s in cp])

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
embeddings_W = tf.Variable(embedding_matrix)

# Input
x1 = tf.placeholder(tf.int32, [None, None])
x2 = tf.placeholder(tf.int32, [None, None])
y = tf.placeholder(tf.float64, [None])
lr = tf.placeholder(tf.float64)

def sentence_embedding(xs):
    xs_mask = 1 - tf.to_double(tf.equal(xs, pad_word_id))
    xs_len = tf.reduce_sum(xs_mask, axis=1)
    xs_embedded = tf.gather(embeddings_W, xs)
    xs_center = tf.reduce_sum(xs_embedded, axis=1) / tf.reshape(tf.to_double(xs_len)+1e-6, [-1, 1])
    return xs_center

x1_center = sentence_embedding(x1)
x2_center = sentence_embedding(x2)
W = tf.Variable(tf.truncated_normal([emb_size, emb_size], stddev=0.01, dtype=tf.float64))
tf_score = tf.reduce_sum((x2_center * (x1_center @ W)), axis=1)

tf_prob = tf.sigmoid(tf_score)
tf_correct = tf.reduce_sum(tf.cast(
    (tf.equal(y, 1) & tf.greater_equal(tf_prob, 0.5)) | (tf.equal(y, 0) & tf.less(tf_prob, 0.5)),
    tf.int32
))


reg = tf.nn.l2_loss(W) * 1e-9
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf_score))
cost_reg = cost + reg
optimizer = tf.train.AdamOptimizer(lr)
gvs = optimizer.compute_gradients(cost_reg)
capped_gvs = [(tf.clip_by_norm(grad, 2), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def eval_valid_loss():
    valid_loss = 0
    valid_acc = 0
    valid_batch = 1024
    batch_num = valid_data_loader.data_num // valid_batch
    for i in range(batch_num):
        b_x1, b_x2, b_y = valid_data_loader.next_batch(valid_batch, max_seq_len, pad_word_id)
        now_loss, now_correct = sess.run([cost, tf_correct], {x1: b_x1, x2: b_x2, y: b_y})
        valid_loss += now_loss / batch_num
        valid_acc += now_correct / (batch_num * valid_batch)
    return valid_loss, valid_acc


learning_rate = 1e-4
batch_size = 256
epoch_num = 40
log_interval = 500
save_interval = 10000

last_epoch = None
train_batch_loss = 0
start_time = time.time()
best_acc = None
for i_batch in range(epoch_num * train_data_loader.data_num // batch_size):
    epoch = i_batch // (train_data_loader.data_num // batch_size)
    if last_epoch is None or last_epoch != epoch:
        last_epoch = epoch
        print('Start epoch %d' % (epoch))

    epoch = i_batch // (train_data_loader.data_num // batch_size)
    b_x1, b_x2, b_y = train_data_loader.next_batch(batch_size, max_seq_len, pad_word_id)
    _, now_loss = sess.run([train_step, cost], {x1: b_x1, x2: b_x2, y: b_y, lr: learning_rate})
    train_batch_loss += now_loss / log_interval
    if (i_batch+1) % log_interval == 0:
        valid_loss, valid_acc = eval_valid_loss()
        print('train batch loss %10f / valid loss %10f / valid acc %10f / elapsed time %.f' % (
            train_batch_loss, valid_loss, valid_acc, time.time()-start_time), flush=True)
        train_batch_loss = 0
        if best_acc is None or best_acc < valid_acc:
            best_acc = valid_acc
            print('model saved (best)', flush=True)
            saver.save(sess, 'models/Attack-sentence-embedding/best')
        # else:
            # learning_rate /= 1.01
            # print('Decay learing rate -> %10f' % (learning_rate))
    if save_interval is not None and (i_batch+1) % save_interval == 0:
        saver.save(sess, 'models/Attack-sentence-embedding/latest')
        print('model saved (latest)', flush=True)

saver.save(sess, 'models/Attack-sentence-embedding/final')











