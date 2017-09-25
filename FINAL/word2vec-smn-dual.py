
# coding: utf-8

# In[2]:

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
    'word2vec/fine-tuned-2',
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

word_vec_models = [KeyedVectors.load(mn) for mn in model_names]
mode_ans = []
for q, rs in zip(test_x1, test_x2):
    now_ans = []
    word_vectors = word_vec_models[0]
    now_ans.extend([
        attack_naive_centroid([q], [rs])[0],
        attack_naive_dis_centroid([q], [rs])[0],
        attack_naive_w_centroid([q], [rs])[0],
    ])
    word_vectors = word_vec_models[1]
    now_ans.extend([
        attack_naive_centroid([q], [rs])[0],
        attack_naive_dis_centroid([q], [rs])[0],
        attack_naive_w_centroid([q], [rs])[0],
    ])
    moder = stats.mode(now_ans)
    if moder.count[0] <= 3:
        word_vectors = word_vec_models[2]
        now_ans = [
            attack_naive_centroid([q], [rs])[0],
            attack_naive_dis_centroid([q], [rs])[0],
            attack_naive_w_centroid([q], [rs])[0],
        ]
        mode_ans.append(stats.mode(now_ans).mode[0])
    else:
        mode_ans.append(moder.mode[0])


# In[14]:

with open(output_fname, 'w') as f:
    f.write('id,ans\n')
    f.write('\n'.join(['%d,%d' % (i+1, a) for i, a in enumerate(mode_ans)]))
    f.write('\n')


# In[ ]:



