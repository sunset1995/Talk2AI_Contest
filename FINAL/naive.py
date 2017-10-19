
# coding: utf-8

# # naive word2vec

# In[1]:

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
from pandas import Series, DataFrame

# Import util
import time
import re
import sys


# ### Load datasets

# In[2]:

input_fname = sys.argv[1]


# In[3]:

test_datas = pd.read_csv(input_fname)
test_x1 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.dialogue.values]
test_x1 = [[[word for word in jieba.cut(s) if word.strip()] for s in q] for q in test_x1]
test_x2 = [[s for s in re.sub('[A-Z]:', '\t', _).split('\t') if len(s.strip())] for _ in test_datas.options.values]
test_x2 = [[[word for word in jieba.cut(s) if word.strip()] for s in rs] for rs in test_x2]


# ### word2vec model

# In[4]:

model_names = [
    'models/word2vec/dual-lstm-12-best',
    'models/word2vec/dual-lstm-12-newest',
    'models/word2vec/dual-lstm-13-best',
    'models/word2vec/dual-lstm-13-newest',
    'models/word2vec/dual-lstm-14-best',
    'models/word2vec/dual-lstm-14-newest',
    'models/word2vec/dual-lstm-15-best',
    'models/word2vec/dual-lstm-15-newest',
    'models/word2vec/dual-lstm-16-best',
    'models/word2vec/dual-lstm-16-newest',
    'models/word2vec/dual-lstm-17-best',
    'models/word2vec/dual-lstm-17-newest',
    'models/word2vec/dual-lstm-18-best',
    'models/word2vec/dual-lstm-18-newest',
    'models/word2vec/dual-lstm-22-best',
    'models/word2vec/dual-lstm-22-newest',
    'models/word2vec/dual-lstm-24-best',
    'models/word2vec/dual-lstm-24-newest',
    'models/word2vec/smn-1-best',
    'models/word2vec/smn-1-newest',
]


# In[5]:

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


# In[ ]:

for mn in model_names:
    word_vectors = KeyedVectors.load(mn)
    now_ans = attack_naive_w_centroid(test_x1, test_x2)
    with open('__naive_'+mn.split('/')[2]+'.txt', 'w') as f:
        f.write(','.join([str(a) for a in now_ans]))

