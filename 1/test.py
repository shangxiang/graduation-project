import jieba.posseg
import jieba
import jieba.analyse
from gensim.models import word2vec
import numpy as np

file  = open('a.txt','r',encoding='utf-8')
words = []
model = word2vec.Word2Vec.load('.\\model\\size20.model')

content = file.read().split('\n')
for i in content:
    if i in model.wv.vocab:
        words.append(model[i])

words = np.array(words)
print(words.shape)
result=[]

for i in range(20):
    newarray = words[:,i]
    newarray = newarray.reshape((1,-1))
    cov = np.cov(newarray)
    result.append(cov)

result = np.array(result).reshape((1,-1))


writefile = open('aresult.txt','w',encoding='utf-8')
for i in result[0]:
    writefile.write("%f\n"%i)


