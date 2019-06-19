from gensim.models import word2vec
import numpy as np
import matplotlib.pyplot as plt
'''
plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)



filex = open('stantard.txt','r',encoding='utf-8')
lines = filex.read().split('\n')
standard = []
for line in lines:
    words = line.split(' ')
    standard.append(float(words[1]))

for i in range(1,128):
    plt.scatter(i,standard[i])

plt.savefig("selfsimilar.png")

plt.show()
'''
'''
model = word2vec.Word2Vec.load(".\\modelxiao\\size50.model")
print(model.similarity(u'哀伤',u'美国'))
print('-----------------分割线----------------------------')
for key in model.most_similar(u'哀伤',topn=20):
    print(key[0],key[1])
'''
'''
plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)

x=np.arange(3,128,1)
y=[]

for k in range(3,128):
    file = open("synset1.txt",'r',encoding='utf-8')
    content = file.read().split('\n')
    label=[]
    data=[]

    for line in content:
        words = line.split(' ')
        label.append(words[0])
        data.append(words[1:])

    file.close()
    count =0
    model = word2vec.Word2Vec.load('.\\modelxiao\\size%d.model'%k)
    for i in range(0,100):
        for key in model.most_similar(label[i], topn=10):
            if key[0] in data[i]:
                count+=1
    print(k,count)
    y.append(count)

y=np.array(y)
plt.plot(x,y,'r',marker='o',label='127标准')

plt.savefig('result127biaozhun啊士大夫.png')
plt.show()
'''
model = word2vec.Word2Vec.load(".\\model\\size100.model")
print(len(model.wv.vocab))