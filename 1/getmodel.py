from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
model = word2vec.Word2Vec.load("size20.model")
print("导入model完成！")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

file = open("vocab_size_20",'r',encoding='utf-8')

data=[]
label=[]
count = 0
lines = file.read().split('\n')
for line in lines:
    list = line.split(' ')
    label.append(list[0])
    for i in range(len(list)-1):
        data.append(float(list[i+1]))

data=np.array(data,dtype=float)

data = data.reshape((-1,20))
print("数据读取完成！准备pca")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tsne = TSNE(n_components=2, init='pca', random_state=0)

data2 = tsne.fit_transform(data)
print("降维完成！")
for i in range(100):
    plt.scatter(data2[i][0],data2[i][1])
    plt.annotate(label[i],xy=(data2[i][0],data2[i][1]),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')

plt.show()
""""
for key in model.most_similar(u'中国',topn=20):
    x,y= model[key[0]]
    plt.scatter(x,y)
    plt.annotate(key[0],xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')




print('-----------------分割线----------------------------')

print(model['经理'])
print('-----------------分割线----------------------------')
print(model['助理'])
print('-----------------分割线----------------------------')
print(model.similarity(u'经理',u'助理'))
print('-----------------分割线----------------------------')
print(model.similarity(u'经理',u'总经理'))
print('-----------------分割线----------------------------')
print(model.similarity(u'经济学家',u'经济学'))
print('-----------------分割线----------------------------')
print(model.similarity(u'中国',u'美国'))
print('-----------------分割线----------------------------')
for key in model.most_similar(u'中国',topn=20):
    print(key[0],key[1])
"""""