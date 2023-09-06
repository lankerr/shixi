# coding:utf-8
import hanlp


def myhanlp(sentence):
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    sen_tok = tok(sentence)

    # 加载预训练模型进行命名实体识别任务
    ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)

    # 命名实体识别结果
    myner = ner([sen_tok], tasks='ner*')

    # 需要处理成含有

    ls = []  # 按每个字给分类

    myner = myner[0]

    # 我收购了，直接暴力匹配吧！
    ls = []
    for i in range(len(sentence)):
        ls.append('0');
    j = 0
    for i in range(len(sentence)):
        if (j >= len(myner)):
            break
        mylen = len(myner[j][0])
        if sentence[i:i + mylen] == myner[j][0]:
            for k in range(mylen):
                ls[i + k] = myner[j][1]
            j += 1

    return ls
