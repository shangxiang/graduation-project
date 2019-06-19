from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)

x=np.arange(3,128)
y10=[]
y8=[]
y6=[]
y4=[]

for k in range(3,128):
    file = open("synset.txt",'r',encoding='utf-8')
    content = file.read().split('\n')
    label=[]
    data=[]

    for line in content:
        words = line.split(' ')
        label.append(words[0])
        data.append(words[1:])

    file.close()
    file = open('synset.txt','w',encoding='utf-8')


    count =[0,0,0,0]
    model = word2vec.Word2Vec.load('.\\modelxiao\\size%d.model'%k)
    for i in range(0,100):
        file.write('%s'%label[i])
        flag=0
        for key in model.most_similar(label[i], topn=10):
            file.write(' %s' % key[0])
            if flag<4:
                if key[0] in data[i][0:4]:
                    count[0]+=1
            if flag<6:
                if key[0] in data[i][0:6]:
                    count[1]+=1
            if flag<8:
                if key[0] in data[i][0:8]:
                    count[2]+=1
            if flag<10:
                if key[0] in data[i]:
                    count[3]+=1
            flag+=1
        file.write('\n')
    file.close()
    y10.append(count[3])
    y8.append(count[2])
    y6.append(count[1])
    y4.append(count[0])
    print(k,count)

y10=np.array(y10)
y8=np.array(y8)
y6=np.array(y6)
y4=np.array(y4)

plt.plot(x,y10,color='r',marker='o',label = '最相似10词')
plt.plot(x,y8,color='g',marker='o',label = '最相似8词')
plt.plot(x,y6,color='y',marker='o',label = '最相似6词')
plt.plot(x,y4,color='b',marker='o',label = '最相似4词')

plt.savefig('resultall.png')
plt.show()

