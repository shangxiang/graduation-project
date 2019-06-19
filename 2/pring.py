import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)

x=[20,40,60,80,100]
y8=[0.1,0.4,0.317,0.383,0.16]
y6=[0.05,0.525,0.384,0.312,0.13]
y4=[-0.15,0.55,0.417,0.287,0.17]

x=np.array(x)
y8=np.array(y8)
y6=np.array(y6)
y4=np.array(y4)

plt.plot(x,y8,color='r',marker='o')
plt.plot(x,y6,color='y',marker='o')
plt.plot(x,y4,color='b',marker='o')

plt.savefig('对比.png')
plt.show()