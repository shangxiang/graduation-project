import numpy as np
import trainchange
import matplotlib.pyplot as plt
file = open('50single.txt','w',encoding='utf-8')

for i in range(1,5):
    accfile = open('result50_single%d0000.txt'%i, 'r', encoding='utf-8')
    lines = accfile.read().split('\n')
    acclist = []
    for j in lines:
        words = j.split(' ')
        acclist.append(float(words[0]))
    acclist = np.array(acclist)
    index = acclist.argsort()
    #index = index[::-1]
    for j in index:
        file.write('%d '%j)
    file.write('\n')



