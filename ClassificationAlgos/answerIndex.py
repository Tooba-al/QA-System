import pandas as pd
from colorama import Fore


def check_for_answer_index(file_name):
    df = pd.read_csv("ClassificationAlgos/" + file_name + ".csv", encoding="utf8")
    answers_list = df["answer"].tolist()
    spans_list = df["predict_span"].tolist()
    index_dic = {}
    count = 0
    for ans_index in range(len(answers_list)):
        # print(ans_index)
        ans = answers_list[ans_index]
        span = spans_list[ans_index]
        newAns = ans.split()
        index_word_dic = {}
        flag = 0

        if ans in span:
            index_dic[ans + str(ans_index)] = span.index(ans)
            flag = 1

        else:
            for word in newAns:
                if word in span:
                    index_word_dic[word] = span.index(word)
                    flag = 1

            if flag:
                index_dic[ans + str(ans_index)] = index_word_dic
            elif not flag:
                index_dic[ans + str(ans_index)] = {-1}

    df.insert(loc=7, column="answer_index(es)", value=list(index_dic.values()))
    df.to_csv("ClassificationAlgos/ansIndex3_NB_dev1.csv", index=False)


check_for_answer_index("predictSpan_NB_dev1")
