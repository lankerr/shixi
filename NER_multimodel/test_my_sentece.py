from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from myhanlp import myhanlp

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记


# cut_sentence函数用于将输入的句子截断，按照。！？截断
def cut_sentence(sentence):
    ls = []
    tmp = ''
    for i in range(len(sentence)):
        if sentence[i] != '。' and sentence[i] != '！' and sentence[i] != '？':
            tmp += sentence[i]
        else:
            tmp += sentence[i]
            ls.append(tmp)
            tmp = ''
    return ls


def print_result(sentence, pred):
    for tmp1, tmp2 in zip(sentence, pred):
        for a, b in zip(tmp1, tmp2):
            print(a, "|", b, end=" ")
        print("\n")


def eval(sentence):
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")

    print("加载并评估hmm模型...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(sentence,
                              word2id,
                              tag2id)
    print("结果：")
    print_result(sentence, hmm_pred)

    #    metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    #    metrics.report_confusion_matrix()  # 打印混淆矩阵

    # 加载并评估CRF模型
    print("加载并评估crf模型...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(sentence)
    print("结果：")
    print_result(sentence, crf_pred)
    #   metrics.report_confusion_matrix()

    # bilstm+hmm模型
    print("加载并评估bilstm+hmm模型...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(sentence, hmm_pred,
                                                   bilstm_word2id, bilstm_tag2id)
    print("结果：")
    print_result(sentence, lstm_pred)
    # metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    # metrics.report_scores()
    #  metrics.report_confusion_matrix()

    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    sentence_list = []
    for senc in sentence:
        tmp = []
        for word in senc:
            tmp.append(word)
        sentence_list.append(tmp)

    sentence_list, test_tag_lists = prepocess_data_for_lstmcrf(
        sentence_list, crf_pred, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(sentence_list, crf_pred,
                                                      crf_word2id, crf_tag2id)

    print("结果：")
    print_result(sentence, lstmcrf_pred)

    mystr = []
    for sec in sentence:
        tmp = myhanlp(sec)
        mystr.append(tmp)
    print("hanlp结果：")
    print_result(sentence, mystr)

    print("天天开心！")
    # metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    # metrics.report_scores()
    #    metrics.report_confusion_matrix()


# 命名实体识别结果

#   ensemble_evaluate(
#      [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
#     test_tag_lists
# )


def main():
    mysentence = input("请输入句子：,输入exit退出\n")
    mysentence = cut_sentence(mysentence)
    print("正在计算...")
    eval(mysentence)


if __name__ == "__main__":
    main()
