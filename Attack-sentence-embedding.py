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
sample_rate_on_training_datas = 1
extra_words = ['<pad>']
unknown_word = None

word2id, id2word, word_p, embedding_matrix, corpus, corpus_id = extractor(word2vec_fname, corpus_fnames, sample_rate_on_training_datas, extra_words, unknown_word)



voc_size = embedding_matrix.shape[0]
emb_size = embedding_matrix.shape[1]
pad_word_id = word2id['<pad>']
max_seq_len = np.max([len(s) for cp in corpus_id for s in cp])

print('%20s: %d' % ('pad_word_id', pad_word_id))
print('%20s: %d' % ('max_seq_len', max_seq_len))

# Data split
rnd_idx = np.arange(len(corpus_id))
np.random.shuffle(rnd_idx)
corpus_id = corpus_id[rnd_idx]
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
y = tf.placeholder(tf.int32, [None])
lr = tf.placeholder(tf.float64)

# Sentence embedding
x1_mask = tf.to_double(tf.not_equal(x1, pad_word_id))
x2_mask = tf.to_double(tf.not_equal(x2, pad_word_id))
x1_len = tf.reduce_sum(x1_mask, axis=1)
x2_len = tf.reduce_sum(x2_mask, axis=1)
x1_embedded = tf.gather(embeddings_W, x1) * tf.reshape(x1_mask, [-1, tf.shape(x1)[1], 1])
x2_embedded = tf.gather(embeddings_W, x2) * tf.reshape(x2_mask, [-1, tf.shape(x2)[1], 1])
x1_center = tf.reduce_sum(x1_embedded, axis=1) / tf.reshape(tf.to_double(x1_len)+1e-6, [-1, 1])
x2_center = tf.reduce_sum(x2_embedded, axis=1) / tf.reshape(tf.to_double(x2_len)+1e-6, [-1, 1])

W = tf.Variable(tf.truncated_normal([emb_size, emb_size], stddev=0.01, dtype=tf.float64))
tf_score = tf.reduce_sum((x2_center * (x1_center @ W)), axis=1)

tf_prob = tf.sigmoid(tf_score)
tf_guess = tf.cast(tf.greater_equal(tf_prob, 0.5), tf.int32)
tf_correct = tf.reduce_sum(tf.cast(tf.equal(y, tf_guess), tf.int32))

reg = tf.nn.l2_loss(W) * 1e-2
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float64), logits=tf_score))
cost_reg = cost + reg
optimizer = tf.train.AdamOptimizer(lr)
gvs = optimizer.compute_gradients(cost_reg)
capped_gvs = [(tf.clip_by_norm(grad, 20), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def eval_valid_loss():
    valid_loss = 0
    reg_loss = 0
    valid_acc = 0
    valid_batch = 2048
    num = [0, 0]
    correct = [0, 0]
    batch_num = valid_data_loader.data_num // valid_batch
    for i in range(batch_num):
        b_x1, b_x2, b_y = valid_data_loader.next_batch(valid_batch, max_seq_len, pad_word_id)
        now_loss, now_reg_loss, now_correct, now_guess = sess.run([cost, reg, tf_correct, tf_guess], {x1: b_x1, x2: b_x2, y: b_y})
        assert(now_correct == np.sum(now_guess == b_y))
        valid_loss += now_loss / batch_num
        reg_loss += now_reg_loss / batch_num
        valid_acc += now_correct / (batch_num * valid_batch)
        num[0] += np.sum(b_y == 0)
        num[1] += np.sum(b_y == 1)
        correct[0] += np.sum((b_y == 0) & (now_guess == b_y))
        correct[1] += np.sum((b_y == 1) & (now_guess == b_y))
    recall_0 = correct[0] / num[0] if num[0] else 0
    recall_1 = correct[1] / num[1] if num[1] else 0
    return valid_loss, reg_loss, valid_acc, recall_0, recall_1


learning_rate = 1e-3
batch_size = 256
epoch_num = 40
log_interval = 1
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
        valid_loss, reg_loss, valid_acc, recall_0, recall_1 = eval_valid_loss()
        print('train batch loss %8f / valid loss %8f / valid reg loss %8f / valid acc %8f / recall_0 %8f / recall_1 %8f / elapsed time %.f' % (
            train_batch_loss, valid_loss, reg_loss, valid_acc, recall_0, recall_1, time.time()-start_time), flush=True)
        train_batch_loss = 0
        if best_acc is None or best_acc < valid_acc:
            best_acc = valid_acc
            print('model saved (best)', flush=True)
            saver.save(sess, 'models/Attack-sentence-embedding-6/best')
        else:
            learning_rate /= 1.01
            print('Decay learing rate -> %10f' % (learning_rate))
    if save_interval is not None and (i_batch+1) % save_interval == 0:
        saver.save(sess, 'models/Attack-sentence-embedding-6/latest')
        print('model saved (latest)', flush=True)

saver.save(sess, 'models/Attack-sentence-embedding-6/final')










