import random
import math
import numpy as np



for k in range(128):
    file = open('stantardcos.txt', 'a', encoding='utf-8')
    sum = 0.0
    for i in range(500000):
        a = []
        b = []

        for j in range(k):
            a.append(random.uniform(0, 1))
            b.append(random.uniform(0, 1))
        m = np.array(a)
        n = np.array(b)
        num=0.0
        for t in range(k):
            num+=a[t]*b[t]
        cos = num/(np.linalg.norm(m)*np.linalg.norm(n))
        sum += (0.5+0.5*cos)

    file.write('%d %f\n'%(k,sum/500000))
    file.close()
    print(k,sum / 500000)


