from gensim.corpora import WikiCorpus
import time


# Extract edu_duct
edu_dict = set()
with open('datas/dict/edu_dict.txt', 'r') as f:
    edu_dict.update([line.strip('\n') for line in f])


# Extract articles + convert to traditional chinese

# !! Warning !!
# Below code will replace original result and run roughly 20 minutes
wiki_corpus = WikiCorpus('datas/raw/zhwiki-20170801-pages-articles.xml.bz2', dictionary=edu_dict)
with open('datas/wiki-texts.txt', 'w', encoding='utf-8') as output:
    start_time = time.time()
    for i, text in enumerate(wiki_corpus.get_texts()):
        output.write(' '.join(text) + '\n')
        if i % 1000 == 0:
            print('Finished %3dk lines / elapsed time %10.2f' % (i/1000, time.time() - start_time), end='\r')
