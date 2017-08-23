from gensim.models import word2vec

corpus_fname = [
    'datas/training_data/下課花路米.txt',
    # 'datas/training_data/人生劇展.txt',
    'datas/training_data/公視藝文大道.txt',
    # 'datas/training_data/成語賽恩思.txt',
    # 'datas/training_data/我的這一班.txt',
    'datas/training_data/流言追追追.txt',
    'datas/training_data/聽聽看.txt',
    'datas/training_data/誰來晚餐.txt',
]


corpus = []
for fname in corpus_fname:
    with open(fname, 'r') as f:
        corpus.extend([[word for s in line.strip().split('\t') for word in s.strip().split()] for line in f])
print('corpus num:', len(corpus))


# Train word2vec model
word2vec_model = word2vec.Word2Vec(corpus, size=200, window=5, workers=4, min_count=5, sample=1e-4, negative=10, iter=15)
print('training time:', word2vec_model.total_train_time)
print('vocab    size:', len(word2vec_model.wv.vocab))

# Saving model
word2vec_model.save('models/official_no_drama_word2vec_200.model.bin')
