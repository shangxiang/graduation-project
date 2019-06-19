# -*-coding:utf-8 -*-
import jieba.analyse
import jieba
import os
# 添加专有名词，增加分词力度

jieba.suggest_freq('中国社科院研究生院', True)
jieba.suggest_freq('德国ZF集团', True)
jieba.suggest_freq('技术换市场', True)
jieba.suggest_freq('中央企业', True)
jieba.suggest_freq('工作会议', True)
jieba.suggest_freq('国资委主任', True)

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'rb').readlines()]

    return stopwords

stopwords = stopwordslist("stop_word.txt")

print(stopwords)

file = open("new1.txt",'rb')
document = file.read()
document_cut=jieba.cut(document,cut_all=False)
result = ' '.join(document_cut)

writefile = open('result.txt','w',encoding='utf-8')
writefile.write(result)
writefile.close()
file.close()

file = open('result.txt','r',encoding='utf-8')
words = file.read().split(' ')

outstr=''
for word in words:
    if word not in stopwords:
        if word != '\t':
            outstr += word
            outstr += ' '

writefile=open('new_result1.txt','w',encoding='utf-8')
writefile.write(outstr)
file.close()
writefile.close()


""""
def cut_word(raw_data_path, cut_data_path):
    data_file_list = os.listdir(raw_data_path)

    corpus = ''

    temp = 0

    for file in data_file_list:
        with open(raw_data_path + file, 'rb') as f:
            print(temp + 1)

            temp += 1

            document = f.read()

            document_cut = jieba.cut(document, cut_all=False)

            # print('/'.join(document_cut))

            result = ' '.join(document_cut)

            corpus += result

        #  print(result)

    with open(cut_data_path + 'corpus.txt', 'w+', encoding='utf-8') as f:

        f.write(corpus)  # 读取的方式和写入的方式要一致

    stopwords = stopwordslist(stop_word_path)  # 这里加载停用词的路径

    with open(cut_data_path + 'corpus.txt', 'r', encoding='utf-8') as f:

        document_cut = f.read()

        outstr = ''

        for word in document_cut:

            if word not in stopwords:

                if word != '\t':
                    outstr += word

                    outstr += " "

    with open(cut_data_path + 'corpus1.txt', 'w+', encoding='utf-8') as f:

        f.write(outstr)  # 读取的方式和写入的方式要一致


if __name__ == "__main__":
    cut_word(raw_data_path, cut_data_path)
"""""

