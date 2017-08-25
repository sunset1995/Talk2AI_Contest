import gc
import numpy as np
from gensim.models import word2vec


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Referenced from: https://github.com/martin-gorner/tensorflow-rnn-shakespeare/blob/master/my_txtutils.py
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch


'''
Extractor
Word based embedding layer scenario
'''
def extractor(word2vec_fname, corpus_fnames, extra_words=[], unknown_word=None):
    assert(unknown_word is None or unknown_word in extra_words)
    
    # Read word2vec model
    word2vec_model = word2vec.Word2Vec.load(word2vec_fname)


    # Extract word2vec
    word2id = {}
    id2word = [None] * (len(word2vec_model.wv.vocab) + len(extra_words)) 
    embedding_matrix = np.zeros([len(word2vec_model.wv.vocab) + len(extra_words), word2vec_model.layer1_size])
    word_p = np.zeros(len(word2vec_model.wv.vocab) + len(extra_words))
    total_word = np.sum([v.count for v in word2vec_model.wv.vocab.values()])

    for i, word in enumerate(extra_words):
        word2id[word] = i + len(word2vec_model.wv.vocab)
        id2word[i + len(word2vec_model.wv.vocab)] = word

    for k, v in word2vec_model.wv.vocab.items():
        word2id[k] = v.index
        id2word[v.index] = k
        word_p[v.index] = v.count / total_word
        embedding_matrix[v.index] = word2vec_model.wv.word_vec(k)
    
    del(word2vec_model)
    gc.collect()


    # Extract corpus
    corpus = []
    for fname in corpus_fnames:
        with open(fname, 'r') as f:
            corpus.extend([[s.split() for s in line.strip().split('\t')] for line in f])
    
    def s_2_sid(s):
        ret = []
        for word in s:
            if word in word2id:
                ret.append(word2id[word])
            else:
                ret.append(word2id[unknown_word] if unknown_word is not None else -1)
        return ret
    corpus_id = [[s_2_sid(s) for s in c] for c in corpus]


    return word2id, id2word, word_p, embedding_matrix, np.array(corpus), np.array(corpus_id)


'''
MiniBatch Helper
PTT Gossip scenario
'''
class MiniBatch():
    def __init__(self, x1, x2, y):
        '''
        Parameters
            x1: np.array. Containing a list of questions
            x2: np.array. Containing options for each corresponding x1(question)
            y : np.array. Containing a list of int which is the answer for corresponding x1(question)
        Note I:
            x1, x2, y inside or outside the class are referencing to same memory.
            So do all returning result by this class.
            This class won't (hope) do any modification on x1, x2, y
        Note II:
            # of batch in a epoch for sigmoid = # of options
            # of batch in a epoch for cross entropy = # of questions
        '''
        if type(x1) != np.ndarray or type(x2) != np.ndarray or type(y) != np.ndarray:
            raise AssertionError('x1, x2, y should be np.ndarray')
        if len(x1) != len(x2) or len(x1) != len(y):
            raise AssertionError('len(x1), len(x2), len(y) should be the same')
        for i in range(len(x2)):
            if len(x2[i]) != len(x2[0]):
                raise AssertionError('Each element of x2 should be the same length')
        self._x1 = x1
        self._x2 = x2
        self._y = y
        self._sigmoid_pointer = 0
        self._sigmoid_idx_pool = np.array([(i, j) for i in range(len(x2)) for j in range(len(x2[i]))])
        self._entropy_pointer = 0
        self._entropy_idx_pool = np.arange(len(x1))
        np.random.shuffle(self._sigmoid_idx_pool)
        np.random.shuffle(self._entropy_idx_pool)


    def next_batch_4_sigmoid(self, batch_size):
        f = self._sigmoid_pointer
        t = self._sigmoid_pointer + batch_size
        if t > len(self._sigmoid_idx_pool):
            f = 0
            t = batch_size
            np.random.shuffle(self._sigmoid_idx_pool)
        self._sigmoid_pointer = t
        idx = self._sigmoid_idx_pool[f:t]
        idx_0 = idx[:, 0]
        idx_1 = idx[:, 1]
        return self._x1[idx_0], self._x2[idx_0, idx_1], np.array(self._y[idx_0]==idx_1, dtype=np.int8)


    def next_batch_4_cross_entropy(self, batch_size):
        f = self._entropy_pointer
        t = self._entropy_pointer + batch_size
        if t > len(self._entropy_idx_pool):
            f = 0
            t = batch_size
            np.random.shuffle(self._entropy_idx_pool)
        self._entropy_pointer = t
        idx = self._entropy_idx_pool[f:t]
        onehot = np.zeros((len(idx), len(x2[0])))
        onehot[np.arange(len(idx)), self._y[idx]] = 1
        return self._x1[idx], self._x2[idx], onehot



'''
MiniBatch Helper
Official corpus scenario
'''
class MiniBatchCorpus():
    def __init__(self, corpus, n_wrong=1):
        '''
        Parameters:
            corpus : list of corpus (2D)
            n_wrong: int. # of wrong answer to be generated for each question.
        Note I:
            This class will create a flatten (1D) version of corpus for convenient.
            But still a reference to outside corpus, changing corpus outside will 
            changing corpus inside the class also.
        '''
        self._corpus = np.array([s for c in corpus for s in c])
        self._pointer = 0
        
        border_idx = np.cumsum([len(c) for c in corpus]) - 1
        que_idx = np.delete(np.arange(np.sum([len(c) for c in corpus])), border_idx)
        ans_idx = que_idx + 1
        
        self._dt_pool = np.vstack([
            np.stack([que_idx, ans_idx, np.ones(len(que_idx), dtype=np.int32)], axis=1),
            *[
                np.stack([que_idx, self.__get_wrong_idx(ans_idx), np.zeros(len(que_idx), dtype=np.int32)], axis=1)
                for i in range(n_wrong)
            ]
        ])
        np.random.shuffle(self._dt_pool)
        
        self.data_num = len(self._dt_pool)


    def __get_wrong_idx(self, ans_idx):
        '''
        Generate a sequence which is a shuffle version of input ans.
        Each output elements is different from input ans.
        '''
        assert(len(ans_idx) > 1)
        idx = ans_idx.copy()
        np.random.shuffle(idx)
        for i in np.where(idx == ans_idx)[0]:
            if idx[i] != ans_idx[i]:
                continue
            t = np.random.randint(len(ans_idx))
            while t==i or idx[i]==ans_idx[t] or idx[t]==ans_idx[i]:
                t = np.random.randint(len(ans_idx))
            idx[i], idx[t] = idx[t], idx[i]
        return idx


    def __padding(self, lst, pad_to_length, pad_word):
        if pad_to_length < len(lst):
            raise Exception('Padding error!! %d > %d' % (len(lst), pad_to_length))
        return [lst[i] if i<len(lst) else pad_word for i in range(pad_to_length)]


    def next_batch(self, batch_size, pad_to_length=-1, pad_word=-1):
        f = self._pointer
        t = self._pointer + batch_size
        if t > self.data_num:
            f = 0
            t = batch_size
            np.random.shuffle(self._dt_pool)
        self._pointer = t
        dt = self._dt_pool[f:t]
        x1 = [list(lst) for lst in self._corpus[dt[:, 0]]]
        x2 = [list(lst) for lst in self._corpus[dt[:, 1]]]
        y = dt[:, 2].copy()
        if pad_to_length > 0:
            for i in range(len(x1)):
                x1[i] = self.__padding(x1[i], pad_to_length, pad_word)
                x2[i] = self.__padding(x2[i], pad_to_length, pad_word)
        return x1, x2, y

    