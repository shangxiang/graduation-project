from gensim.models import word2vec


writefile = open('synset.txt','w',encoding='utf-8')
model = word2vec.Word2Vec.load('.\\modelxiao\\size3.model')
count=0
for word in model.wv.vocab:
    if count>=100:
        break
    else:
        count+=1
    writefile.write('%s'%word)
    for key in model.most_similar(word, topn=10):
        writefile.write(' %s'%key[0])
    writefile.write('\n')

writefile.close()