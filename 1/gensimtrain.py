from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(r'result1.txt')
for i in range(3,128):
    model = word2vec.Word2Vec(sentences,size=i,hs=1,min_count=10,window=3)
    model.save(".\\modelxiao\\size%d.model"%i)
    print("modelsize=%d完成！"%i)
