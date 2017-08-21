import numpy as np


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


class mini_batcher_corpus():
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


    def next_batch(self, batch_size):
        f = self._pointer
        t = self._pointer + batch_size
        if t > self.data_num:
            f = 0
            t = batch_size
            np.random.shuffle(self._dt_pool)
        self._pointer = t
        dt = self._dt_pool[f:t]
        x1 = self._corpus[dt[:, 0]]
        x2 = self._corpus[dt[:, 1]]
        y = dt[:, 2]
        return x1, x2, y