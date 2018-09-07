from  dataPreprocess import *
import re
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from fastText import train_supervised

if __name__ == '__main__':
    SogouTCE_kv=load_SogouTCE()
    
    #text_split()
    #labels=load_url(SogouTCE_kv)

    x,y=load_selecteddata(SogouTCE_kv)

    stopwords=load_stopwords()

    #切割token
    x=[  [word for word in line.split() if word not in stopwords]   for line in x]

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #按照fasttest的要求生成训练数据和测试数据
    dump_file(x_train,y_train,root + "test/fasttext/sougou_train.txt")
    dump_file(x_test, y_test, root + "test/fasttext/sougou_test.txt")

    import ChineseTextClassifyTool.fasttext as ft
    # train_supervised uses the same arguments and defaults as the fastText cli
    model = ft.fasttext(root + "test/fasttext/sougou_train.txt")
    print_results(*model.test(root + "test/fasttext/sougou_test.txt"))