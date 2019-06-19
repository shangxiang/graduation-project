import numpy as np
import matplotlib.pyplot as plt
from gensim.models import word2vec
import math

plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)

for k in range(3,128):
    model = word2vec.Word2Vec.load(".\\modelxiao\\size%d.model"%k)

    data = model.wv.vectors
    print(data.shape)
    data = data.transpose()
    cov = np.cov(data)
    eigenvalue,featurevector=np.linalg.eig(cov)
    #print(cov.shape)

    sum=0.0
    list=[]

    for i in range(0,k):
        list.append(data[i][i])

    result = np.array(list)
    dx = np.cov(result)

    plt.scatter(k,dx)
    print(k,dx)

plt.savefig('对角线方差.png')
plt.show()