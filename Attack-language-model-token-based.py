import numpy as np
import tensorflow as tf

# Import util
import time
import re
import sys
import gc

# Self define module
from mini_batch_helper import extractor
from mini_batch_helper import rnn_minibatch_sequencer



word2vec_fname = 'models/word2vec_all_offitial_200.model.bin'
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
sample_rate_on_training_datas = 0.3
extra_words = ['<unk>', '<bos>', '<eos>']
unknown_word = '<unk>'

word2id, id2word, word_p, embedding_matrix, corpus, corpus_id = extractor(word2vec_fname, corpus_fnames, sample_rate_on_training_datas, extra_words, unknown_word)
del(word_p)
del(embedding_matrix)
del(corpus)

# Get only fixed number of corpus
rnd_idx = np.arange(len(corpus_id))
np.random.shuffle(rnd_idx)
corpus_id = corpus_id[rnd_idx[:100]]

train_corpus_id = corpus_id[:len(corpus_id)-1]
valid_corpus_id = corpus_id[len(corpus_id)-1:]
traintext = [w for cp in train_corpus_id for s in cp for w in [word2id['<bos>']] + s + [word2id['<eos>']]]
validtext = [w for cp in valid_corpus_id for s in cp for w in [word2id['<bos>']] + s + [word2id['<eos>']]]
del(corpus_id)
del(train_corpus_id)
del(valid_corpus_id)



SEQLEN = 10
BATCHSIZE = 32
EPOCHNUM = 10
ALPHASIZE = len(word2id)
INTERNALSIZE = 200
NLAYERS = 2
LEARNING_RATE = 1e-4
DROPOUT_PKEEP = 1
LOGINTERVAL = 10
SAVEINTERVAL= 100
# CLIP = 0.2

print('%20s: %s' % ('SEQLEN', SEQLEN))
print('%20s: %s' % ('BATCHSIZE', BATCHSIZE))
print('%20s: %s' % ('EPOCHNUM', EPOCHNUM))
print('%20s: %s' % ('ALPHASIZE', ALPHASIZE))
print('%20s: %s' % ('INTERNALSIZE', INTERNALSIZE))
print('%20s: %s' % ('NLAYERS', NLAYERS))
print('%20s: %s' % ('LEARNING_RATE', LEARNING_RATE))
print('%20s: %s' % ('DROPOUT_PKEEP', DROPOUT_PKEEP))
print('%20s: %s' % ('LOGINTERVAL', LOGINTERVAL))
print('%20s: %s' % ('SAVEINTERVAL', SAVEINTERVAL))
# print('%20s: %s' % ('CLIP', CLIP))



# inputs
X = tf.placeholder(tf.int32, [None, None])    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)       # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
Y_ = tf.placeholder(tf.int32, [None, None])   # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]

# inputs info
batchsize = tf.placeholder(tf.int32)

# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS])  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

cells = [tf.contrib.rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=DROPOUT_PKEEP) for cell in cells]
multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=DROPOUT_PKEEP)

# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] (last state in the sequence)
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])               # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = tf.contrib.layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])                # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits)                   # [ BATCHSIZE x SEQLEN, ALPHASIZE ]

# Gradient clipping
# optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
# gvs = optimizer.compute_gradients(loss)
# capped_gvs = [(tf.clip_by_norm(grad, CLIP), var) for grad, var in gvs]
# train_step = optimizer.apply_gradients(capped_gvs)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)



saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

step= 0
start_time = time.time()
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
batch_loss = 0

for x, y_, epoch in rnn_minibatch_sequencer(traintext, BATCHSIZE, SEQLEN, EPOCHNUM):
    step += 1
    _, now_loss, istate = sess.run([train_step, loss, H], {
        X: x,
        Y_: y_,
        Hin: istate,
        batchsize: BATCHSIZE,
    })
    batch_loss += np.mean(now_loss) / LOGINTERVAL
    if step % LOGINTERVAL == 0:
        print('epoch %2d: batch loss %10f / elapsed time %.f' % (epoch, batch_loss, time.time() - start_time), flush=True)
        batch_loss = 0
    if step % SAVEINTERVAL == 0:
        saver.save(sess, 'models/Attack-language-model/lm', global_step=step)
        print('Saved model', flush=True)

saver.save(sess, 'models/Attack-language-model/lm-final')


