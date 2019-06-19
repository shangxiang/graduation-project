import numpy as np
import matplotlib.pyplot as plt
import math
from gensim.models import word2vec

plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)

for k in range(3,128):
    model = word2vec.Word2Vec.load(".\\modelxiao\\size%d.model"%k)

    data = model.wv.vectors

    data = data.transpose()
    cov = np.cov(data)
    #print(cov)
    tem=[]
    sum=0.0

    for i in range(0,k-1):
        for j in range(i+1,k):
            sum+=cov[i][j]/(math.sqrt(cov[i][i]*cov[j][j]))

    mean = sum/(k*(k-1)/2)
    plt.scatter(k,math.fabs(mean))
    print(k,math.fabs(mean))

plt.savefig('rou_mean_abs.png')
plt.show()