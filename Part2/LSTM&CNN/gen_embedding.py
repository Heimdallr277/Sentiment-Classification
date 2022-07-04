from gensim.models import Word2Vec
import numpy as np
# max_sen_len 50 max_doc_len 40
embedding_path = 'data/embedding.txt'
doc = []
sentences = []
doc_len = []
sen_len = []

with open('data/train.ss', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t\t')
        doc.append(line[3].lower())
    f.close()

with open('data/test.ss', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t\t')
        doc.append(line[3].lower())
    f.close()

with open('data/dev.ss', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t\t')
        doc.append(line[3].lower())
    f.close()
words = []
for d in doc:
    sentences = d.split('<sssss>')
    doc_len.append(len(sentences))
    for s in sentences:
        words = s.strip().split()
        sen_len.append(len(words))

embeddings = Word2Vec(words)
embeddings.wv.save_word2vec_format(embedding_path, binary=False)


