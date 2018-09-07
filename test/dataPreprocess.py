root = 'E:/10GitHub/ChineseTextClassifyTool/ChineseTextClassifyTool/'
import re
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def load_stopwords():
    with open(root + "data/stopwords.txt", encoding='utf-8') as F:
        stopwords=F.readlines()
        F.close()
    return [word.strip() for word in stopwords]

def load_selecteddata(SogouTCE_kv):
    x=[]
    y=[]

    #加载content列表
    #with codecs.open(root + "data/news_sohusite_content.txt", "r", encoding='utf-8', errors='ignore') as F:
    with open(root + "data/news_sohusite_content.txt", encoding='utf-8') as F:
        content=F.readlines()
        F.close()

    # 加载url列表
    with open(root + "data/news_sohusite_url.txt", encoding='utf-8') as F:
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

def load_SogouTCE():
    SogouTCE=[]
    SogouTCE_kv = {}
    with open(root + "data/SogouTCE.txt") as F:
        for line in F:
            (url,channel)=line.split()
            SogouTCE.append(url)
        F.close()
    import re
    for index,url in enumerate(SogouTCE):
        #删除http前缀
        url=re.sub('http://','',url)
        print("k:%s v:%d" % (url,index))
        SogouTCE_kv[url]=index

    return  SogouTCE_kv

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


