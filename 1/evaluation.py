from gensim.models import word2vec
import matplotlib.pyplot as plt

plt.figure(figsize=(30,18))
plt.tick_params(axis='both',which='both',labelsize=40)

filex = open('stantard.txt','r',encoding='utf-8')
lines = filex.read().split('\n')
standard = []
for line in lines:
    words = line.split(' ')
    standard.append(float(words[1]))

for i in range(3,128):
    model = word2vec.Word2Vec.load(".\\modelxiao\\size%d.model"%i)
    file = open("Syn.txt",'r',encoding='utf-8')

    lines = file.read().split('\n')

    sum = 0.0
    count=0

    for line in lines:
        words = line.split(' ')
        if words[0] in model.wv.vocab and words[1] in model.wv.vocab:
            sum+=model.similarity(words[0],words[1])
            count+=1
    plt.scatter(i,sum/count-standard[i])
    print(i,'完成！')

plt.savefig("similarMistandard.png")

plt.show()

