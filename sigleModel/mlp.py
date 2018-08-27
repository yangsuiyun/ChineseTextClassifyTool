# -*- coding: UTF-8 -*-
import re
from fastText import train_supervised

import numpy as np
import codecs

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

def load_stopwords():
    with open("stopwords.txt") as F:
        stopwords=F.readlines()
        F.close()
    return [word.strip() for word in stopwords]

def load_SogouTCE():
    SogouTCE=[]
    SogouTCE_kv = {}
    with open("../data/SogouTCE.txt") as F:
        for line in F:
            (url,channel)=line.split()
            SogouTCE.append(url)
        F.close()

    for index,url in enumerate(SogouTCE):
        #删除http前缀
        url=re.sub('http://','',url)
        print("k:%s v:%d" % (url,index))
        SogouTCE_kv[url]=index

    return  SogouTCE_kv

##生成后续需要的两个文件：url和content
def text_split():
    import xml.etree.ElementTree as ET
    tree = ET.parse("../data/news_sohusite_xml.smarty.txt")
    root = tree.getroot()
    
    with open("../data/news_sohusite_url.txt",'w',encoding='utf-8') as F2:
        with open("../data/news_sohusite_content.txt",'w',encoding='utf-8') as F3:
            for doc in root.findall('doc'):
                url = doc.find('url').text
                content = doc.find('content').text
                F2.write(url + '\n')
                F3.write(content)
                F3.write('\n')

def load_url(SogouTCE_kv):
    labels=[]
    with open("../data/news_sohusite_url.txt") as F:
    #with codecs.open("../data/news_sohusite_url.txt","r",encoding='utf-8', errors='ignore') as F:
        for line in F:
            for k,v in SogouTCE_kv.items():
                if re.search(k,line,re.IGNORECASE):
                    #print "x:%s y:%d" % (line,v)
                    print(v)
                    labels.append(v)
                #else:
                #    print "not found %s" %(line)

        F.close()
    return  labels

def load_selecteddata(SogouTCE_kv):
    x=[]
    y=[]

    #加载content列表
    #with codecs.open("../data/news_sohusite_content.txt", "r", encoding='utf-8', errors='ignore') as F:
    with open("../data/news_sohusite_content.txt", encoding='utf-8') as F:
        content=F.readlines()
        F.close()

    # 加载url列表
    with open("../data/news_sohusite_url.txt", encoding='utf-8') as F:
        url = F.readlines()
        F.close()

    for index,u in  enumerate(url):
        for k, v in SogouTCE_kv.items():
            # 只加载id为81，79和91的数据,同时注意要过滤掉内容为空的
            #if re.search(k, u, re.IGNORECASE) and v in (81,79, 91) and len(content[index].strip()) > 1:
            if re.search(k, u, re.IGNORECASE) and len(content[index].strip()) > 1:
                #保存url对应的content内容
                x.append(content[index])
                y.append(v)
                continue

    return x,y


def dump_file(x,y,filename):
    with open(filename, 'w') as f:
        #f.write('Hello, world!')
        for i,v in enumerate(x):
            #f.write("%s __label__%d" % (v,y))
            line="%s __label__%d\n" % (v,y[i])
            #print line
            f.write(line)
        f.close()

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def do_mlp(x,y):

    #mlp
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 3),
                        random_state=1)

    scores = cross_val_score(clf, x, y, cv = 5,scoring='f1_micro')
    #print scores
    print("f1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(clf, x, y, cv = 5,scoring='accuracy')
    #print scores
    print("accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    SogouTCE_kv=load_SogouTCE()

    #labels=load_url(SogouTCE_kv)

    x,y=load_selecteddata(SogouTCE_kv)

    #切割词袋
    vectorizer = CountVectorizer(ngram_range=(2,2),max_features=5000)
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    x = transformer.fit_transform(vectorizer.fit_transform(x))

    #转换成one hot编码
    t=[]
    for i in y:
        if i == 0:
            t.append(0)

        if i == 1:
            t.append(1)

        if i == 2:
            t.append(2)

        if i == 3:
            t.append(3)

    y=to_categorical(t)


    do_mlp(x,y)



