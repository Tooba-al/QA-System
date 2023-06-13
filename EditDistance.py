from nltk.stem import WordNetLemmatizer
import datetime
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))


# Extract Min deletion or Insertion to transfrom question to sentence (Edit Distance)
# anchors_data = pd.read_csv("CSV-Files/ExtractAnchors_dev1.csv")
anchors_data = pd.read_csv("CSV-Files/QS_WordLemma_dev1.csv")
question_list = anchors_data["question"].tolist()
sentence_list = anchors_data["sentence"].tolist()


def stopword_func(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(('?', 'and', 'I', 'A', 'And', 'So', '.', 'as', 'As', '\'\'', 'could', '[', ']', ',', ')', '\'s', 'By', '(', "''",
                       'Other', '``', ':', '\'', '#', '\'v', 'The', ';', 'however', 'still',
                      'the', 'They', 'For', 'also', 'In', 'This', 'When', 'It', 'many', 'Many', 'so', 'cant', 'Yes', 'yes', 'No', 'no',
                       'These', 'these', 'This', 'Where', 'Which', 'Why', 'How', 'What', 'If', 'Who', 'When', 'at'))

    str1 = ""
    for ele in text:
        str1 += ele
    lemmatizer = WordNetLemmatizer()
    word_tokens1 = word_tokenize(str1)
    filtered_sentence = [lemmatizer.lemmatize(
        word) for word in word_tokens1 if not word in set(stop_words)]

    return filtered_sentence


# def editDistance(anchor1, anchor2, anchorQ_index, anchorS_index, Q, S):
def editDistance(Q, S):
    question = stopword_func(Q)
    sentence = stopword_func(S)

    ed = 0

    for word in question:
        if question.count(word) > 0:
            ed += sentence.count(word)
            # if (question[Q_index - 1] == sentence[S_index - 1]):
            #     vector[Q_index][S_index] = vector[Q_index - 1][S_index - 1] + 1

            # else:
            #     vector[Q_index][S_index] = max(vector[Q_index - 1][S_index],
            #                                    vector[Q_index][S_index - 1])

    # L[m][n] contains length of LCS
    # for X[0..n-1] and Y[0..m-1]
    # return vector[len(question)][len(sentence)]
    return (len(question) - ed) + (len(sentence) - ed)


ed_list = []
# questions_list = []
# sentences_list = []
# paragraphNo_list = []
# titleNo_list = []
for index in range(len(sentence_list)):
    ed = editDistance(sentence_list[index], question_list[index])
    # min_QS = editDistance(
    #     anchor_list[index], anchor_question_list[index], anchor_sentence_list[index], sentence_list[index], question_list[index])
    ed_list.append(ed)


min_del_ins_data = {
    "question": question_list,
    "sentence": sentence_list,
    "MIN_DEL_INS": ed_list,
    "paragraphNo": anchors_data["paragraphNo"],
    "titleNo": anchors_data["titleNo"],
}

df = pd.DataFrame(min_del_ins_data)
df.to_csv('CSV-Files/EditDistances_dev1.csv', encoding='utf-8', index=False)

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
