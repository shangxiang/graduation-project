import random
import math
import numpy as np



for k in range(128):
    file = open('stantard.txt', 'a', encoding='utf-8')
    sum = 0.0
    for i in range(500000):
        a = []
        b = []

        for j in range(k):
            a.append(random.uniform(0, 1))
            b.append(random.uniform(0, 1))
        m = np.array(a)
        n = np.array(b)
        c = m - n
        temp = 0
        for item in c:
            temp += item ** 2
        sum += 1/(1+math.sqrt(temp))

    file.write('%d %f\n'%(k,sum/500000))
    file.close()
    print(k,sum / 500000)


