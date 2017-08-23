import numpy as np
import tensorflow as tf
from scipy import spatial

# Import & Init jieba
import jieba
jieba.set_dictionary('datas/dict/dict.txt.big')
jieba.load_userdict('datas/dict/edu_dict.txt')

# Import pandas
import pandas as pd
from pandas import Series, DataFrame

# Import util
import time
import re
import sys
import gc

# Self define module
from mini_batch_helper import MiniBatchCorpus


# Loading sample datas
sample = pd.read_csv('datas/sample_test_data.txt')

# Extract sample test datas
sample_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.dialogue.values]
sample_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in sample.options.values]

# Tokenize
sample_x1 = [list(jieba.cut(' '.join(_))) for _ in sample_x1]
sample_x2 = [[list(jieba.cut(s)) for s in _] for _ in sample_x2]
sample_y = sample.answer.values
assert(np.sum([len(_)!=6 for _ in sample_x2]) == 0)

# Loading test data
test_datas = pd.read_csv('datas/AIFirstProblem.txt')

# Extract test datas
test_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.dialogue.values]
test_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.options.values]
assert(np.sum([len(_)!=6 for _ in test_x2]) == 0)


# Load corpus
corpus_fname = [
    'datas/training_data/下課花路米.txt',
#     'datas/training_data/人生劇展.txt',
#     'datas/training_data/公視藝文大道.txt',
#     'datas/training_data/成語賽恩思.txt',
#     'datas/training_data/我的這一班.txt',
#     'datas/training_data/流言追追追.txt',
#     'datas/training_data/聽聽看.txt',
    'datas/training_data/誰來晚餐.txt',
]

corpus = []
for fname in corpus_fname:
    with open(fname, 'r') as f:
        corpus.extend([[s.split() for s in line.split('\t')] for line in f])

        word2id = {'<PAD>': 0}
id2word = ['<PAD>']
word_p = [0]

# Extract dictionary from corpus
for cp in corpus:
    for sentence in cp:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word.append(word)
                word_p.append(0)
            word_p[word2id[word]] += 1

# Caculate word probability
total_word = np.sum(word_p)
for i in range(len(word_p)):
    word_p[i] /= total_word

# Asserting result
for k, v, in word2id.items():
    assert(id2word[v] == k)
for i in range(1, len(id2word)):
    assert(i == word2id[id2word[i]])
assert(abs(np.sum(word_p)-1) < 1e-9)

data_loader = MiniBatchCorpus(corpus)
max_seq_len = np.max([len(s) for cp in corpus for s in cp])

def word_list_2_id_list(lst):
    return [word2id[lst[i]] if i<len(lst) and lst[i] in word2id else 0 for i in range(max_seq_len)]

sample_id1 = np.array([word_list_2_id_list(q) for q in sample_x1])
sample_id2 = np.array([[word_list_2_id_list(r) for r in rs] for rs in sample_x2])
test_id1 = np.array([word_list_2_id_list(q) for q in test_x1])
test_id2 = np.array([[word_list_2_id_list(r) for r in rs] for rs in test_x2])


# Define model
embedding_size = 200
with tf.device('/cpu:0'):
    tf_word_p = tf.constant(word_p, dtype=tf.float64)
    embeddings_W = tf.Variable(tf.truncated_normal(
        [len(word2id), embedding_size], stddev=1/embedding_size, dtype=tf.float64
    ))

with tf.device('/gpu:0'):
    wa = tf.placeholder(tf.float64, [1])
    x1 = tf.placeholder(tf.int32, [None, max_seq_len])
    x2 = tf.placeholder(tf.int32, [None, max_seq_len])
    y = tf.placeholder(tf.float64, [None])
    x1_mask = tf.to_int32(tf.greater(x1, 0))
    x2_mask = tf.to_int32(tf.greater(x2, 0))
    x1_embedded = tf.gather(embeddings_W, x1*x1_mask)
    x2_embedded = tf.gather(embeddings_W, x2*x2_mask)
    x1_word_p = tf.gather(tf_word_p, x1*x1_mask)
    x2_word_p = tf.gather(tf_word_p, x2*x2_mask)
    x1_len = tf.reduce_sum(x1_mask, axis=1)
    x2_len = tf.reduce_sum(x2_mask, axis=1)
    x1_weighted = tf.reshape(wa / (wa + x1_word_p), [-1, max_seq_len, 1]) * x1_embedded
    x2_weighted = tf.reshape(wa / (wa + x2_word_p), [-1, max_seq_len, 1]) * x2_embedded
    x1_center = tf.reduce_sum(x1_weighted, axis=1) / tf.reshape(tf.to_double(x1_len), [-1, 1])
    x2_center = tf.reduce_sum(x2_weighted, axis=1) / tf.reshape(tf.to_double(x2_len), [-1, 1])
    W = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], stddev=1/embedding_size, dtype=tf.float64))
    tf_score = tf.reduce_sum((x2_center * (x1_center @ W)), axis=1)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf_score))
optimizer = tf.train.AdamOptimizer().minimize(cost)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 2048
epoch_num = 100


sum_cost = 0
start_time = time.time()
for i_batch in range(epoch_num * data_loader.data_num // batch_size):
    b_x1, b_x2, b_y = data_loader.next_batch(batch_size)
    b_x1 = [word_list_2_id_list(q) for q in b_x1]
    b_x2 = [word_list_2_id_list(r) for r in b_x2]
    _, now_cost = sess.run([optimizer, cost], {wa: [1e-4], x1: b_x1, x2: b_x2, y: b_y})
    sum_cost += now_cost
    if (i_batch+1) % 100 == 0:
        now_score = sess.run(tf_score, {
            wa: [1e-4],
            x1: np.repeat(sample_id1, 6, axis=0),
            x2: sample_id2.reshape(-1, max_seq_len),
        })
        now_score = now_score.reshape(-1, 6)
        my_ans = np.argmax(now_score, axis=1)
        sample_correct = np.sum(my_ans == sample_y)
        print('batch cost %10f / sample correct %4d / elapsed time %.f' % (sum_cost, sample_correct, time.time()-start_time))
        sys.stdout.flush()
        sum_cost = 0
        saver.save(sess, 'models/Attack-sentence-embedding/s_emb', global_step=i_batch+1)





