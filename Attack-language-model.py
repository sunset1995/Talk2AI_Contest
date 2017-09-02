import numpy as np
from scipy import spatial
import tensorflow as tf
import pandas as pd

# Import util
import time
import re
import sys
import gc

# Self define module
from mini_batch_helper import rnn_minibatch_sequencer



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
valid_cp_num_of_each = 1

def word_tok_lst_2_ch_lst(s):
    return [ch.strip() for word in s for ch in word if ch.strip() != ''] + ['</s>']

def corpus_flatten(now_corpus):
    return [ch for line in now_corpus for s in line.strip().split('\t') for ch in word_tok_lst_2_ch_lst(s)]

corpus = []
corpus_valid = []
for fname in corpus_fnames:
    with open(fname, 'r') as f:
        now_corpus = np.array([line for line in f])
        now_corpus_valid = now_corpus[:valid_cp_num_of_each]
        now_corpus = now_corpus[valid_cp_num_of_each:]
        if sample_rate_on_training_datas < 1:
            sample_num = int(max(len(now_corpus)*sample_rate_on_training_datas, 5))
            rnd_idx = np.arange(len(now_corpus))
            np.random.shuffle(rnd_idx)
            now_corpus = now_corpus[rnd_idx[:sample_num]]
        
        corpus.extend(corpus_flatten(now_corpus))
        corpus_valid.extend(corpus_flatten(now_corpus_valid))

with open('datas/dict/id2ch.txt') as f:
    id2ch = f.read().strip().split()
ch2id = dict([(ch, i) for i, ch in enumerate(id2ch)])
traintext = np.array([ch2id[ch] for ch in corpus])
validtext = np.array([ch2id[ch] for ch in corpus_valid])
validtext_num = len(validtext)
del(corpus)
del(corpus_valid)

print('%20s: %s' % ('traintext length', len(traintext)))
print('%20s: %s' % ('validtext length', len(validtext)))
print('%20s: %s' % ('vocab size', len(id2ch)))



SEQLEN = 35
BATCHSIZE = 256
EPOCHNUM = 40
ALPHASIZE = len(id2ch)
INTERNALSIZE = 200
EMBEDDINGSIZE = 200      # Must be that EmbeddingSize == INTERNALSIZE
NLAYERS = 3
LEARNING_RATE = 1e-3
LEARNING_DECAY = 1.01
DROPOUT_PKEEP = 0.5
LOGINTERVAL = 500
SAVEINTERVAL= 10000
CLIP = 20

print('%20s: %s' % ('SEQLEN', SEQLEN))
print('%20s: %s' % ('BATCHSIZE', BATCHSIZE))
print('%20s: %s' % ('EPOCHNUM', EPOCHNUM))
print('%20s: %s' % ('ALPHASIZE', ALPHASIZE))
print('%20s: %s' % ('INTERNALSIZE', INTERNALSIZE))
print('%20s: %s' % ('NLAYERS', NLAYERS))
print('%20s: %s' % ('LEARNING_RATE', LEARNING_RATE))
print('%20s: %s' % ('LEARNING_DECAY', LEARNING_DECAY))
print('%20s: %s' % ('DROPOUT_PKEEP', DROPOUT_PKEEP))
print('%20s: %s' % ('LOGINTERVAL', LOGINTERVAL))
print('%20s: %s' % ('SAVEINTERVAL', SAVEINTERVAL))
print('%20s: %s' % ('CLIP', CLIP))



# inputs
X = tf.placeholder(tf.int32, [None, None])    # [ BATCHSIZE, SEQLEN ]
Y_ = tf.placeholder(tf.int32, [None, None])   # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)     # [ BATCHSIZE, SEQLEN, ALPHASIZE ]

# inputs info
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)
batchsize = tf.placeholder(tf.int32)

# embedding layer
rndrng = 6 / (EMBEDDINGSIZE * EMBEDDINGSIZE)
embeddings_w = tf.Variable(
    np.random.uniform(-rndrng, rndrng, [ALPHASIZE, EMBEDDINGSIZE]).astype(np.float32)
)

# input state
Xemb = tf.gather(embeddings_w, X)                               # [ BATCHSIZE, SEQLEN, EMBEDDINGSIZE ]
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS])  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

cells = [tf.contrib.rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] (last state in the sequence)
Yr, H = tf.nn.dynamic_rnn(multicell, Xemb, dtype=tf.float32, initial_state=Hin)

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])               # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Yflat = Yflat @ tf.transpose(embeddings_w)               # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])                # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Yflat, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])                 # [ BATCHSIZE, SEQLEN ]

# Gradient clipping
optimizer = tf.train.AdamOptimizer(lr)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, CLIP), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)



saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def eval_valid_loss():
    valid_seq_len = 2000
    istate = np.zeros([1, INTERNALSIZE*NLAYERS])
    valid_loss = []
    for i in range(0, validtext_num, valid_seq_len):
        nowtext = validtext[i:i+valid_seq_len]
        if len(nowtext) == 1:
            continue
        now_x = [nowtext[:-1]]
        now_y = [nowtext[1:]]
        now_loss, istate = sess.run([loss, H], {
            X: now_x,
            Y_: now_y,
            Hin: istate,
            pkeep: 1,
            batchsize: 1,
        })
        valid_loss.append((np.mean(now_loss), len(now_loss)))
    valid_loss = np.array(valid_loss)
    return np.sum(valid_loss[:, 0] * valid_loss[:, 1]) / np.sum(valid_loss[:, 1])

def generate_text(pre_s, deterministic=True, max_output_len=35):
    pre_s = list(pre_s)
    pre_id = [ch2id[w] for w in pre_s if w in ch2id]
    istate = np.zeros([1, INTERNALSIZE*NLAYERS])  # initial zero input state
    istate = sess.run(H, {X: [pre_id[:-1]], Hin: istate, pkeep: 1})
    now_word_id = pre_id[-1]
    output_lst = []
    while now_word_id != ch2id['</s>'] and len(output_lst) < max_output_len:
        next_word_prob, istate = sess.run([Yflat, H], {X: [[now_word_id]], Hin: istate, pkeep: 1})
        next_word_prob = next_word_prob.astype(np.float64)
        next_word_prob = np.exp(next_word_prob[0]) / np.sum(np.exp(next_word_prob[0]))
        if deterministic:
            next_word_id = np.argmax(next_word_prob)
        else:
            next_word_id = np.argmax(np.random.multinomial(1, next_word_prob))
        output_lst.append(id2ch[next_word_id])
        now_word_id = next_word_id
    return ''.join(output_lst)

def run_validation(valid_text='你'):
    valid_loss = eval_valid_loss()
    print('%20s: %s' % ('Valid loss', valid_loss))
    print('%20s -> %s' % (valid_text, generate_text(valid_text, deterministic=False)))
    return valid_loss

step= 0
start_time = time.time()
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
batch_loss = 0

last_epoch = None
best_valid_loss = None
for x, y_, epoch in rnn_minibatch_sequencer(traintext, BATCHSIZE, SEQLEN, EPOCHNUM):
    if last_epoch is None or last_epoch != epoch:
        last_epoch = epoch
        print('Start epoch %d' % epoch, flush=True)

    step += 1
    _, now_loss, istate = sess.run([train_step, loss, H], {
        X: x,
        Y_: y_,
        Hin: istate,
        lr: LEARNING_RATE,
        pkeep: DROPOUT_PKEEP,
        batchsize: BATCHSIZE,
    })
    batch_loss += np.mean(now_loss) / LOGINTERVAL
    if step % LOGINTERVAL == 0:
        now_valid_loss = run_validation()
        print('batch loss %10f / valid loss %10f / elapsed time %.f' % (
            batch_loss, now_valid_loss, time.time() - start_time), flush=True)
        if best_valid_loss is None or now_valid_loss < best_valid_loss:
            best_valid_loss = now_valid_loss
            saver.save(sess, 'models/Attack-language-model/best')
            print('Saved model (best)', flush=True)
        else:
            LEARNING_RATE /= LEARNING_DECAY
            print('Decayed learning rate: %f' % LEARNING_RATE, flush=True)
        batch_loss = 0
    if step % SAVEINTERVAL == 0:
        saver.save(sess, 'models/Attack-language-model/latest')
        print('Saved model (latest)', flush=True)

saver.save(sess, 'models/Attack-language-model/final')
