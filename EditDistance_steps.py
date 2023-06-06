from nltk.stem import WordNetLemmatizer
import datetime
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv


print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))

lemmatizer = WordNetLemmatizer()

# data = pd.read_csv("CSV-Files/EditDistance_dev1.csv")
# question_list = data["question"].tolist()
# sentence_list = data["sentence"].tolist()


# def word_lemmma(sentence, question):
#     sent_lemma = []
#     ques_lemma = []

#     for word in sentence.split():
#         sent_lemma.append(lemmatizer.lemmatize(word))

#     for word in question.split():
#         ques_lemma.append(lemmatizer.lemmatize(word))

#     edit_distance = 0
#     for word in sent_lemma:
#         if word in ques_lemma:
#             edit_distance += 1

#     return edit_distance


# wordLemma_list = []
# for index in range(len(question_list)):
#     ED = word_lemmma(question_list[index], sentence_list[index])
#     wordLemma_list.append(ED)


# newData = {
#     "question": question_list,
#     "sentence": sentence_list,
#     "word_lemma": wordLemma_list,
#     "paragraphNo": data["paragraphNo"].tolist(),
#     "titleNo": data["titleNo"].tolist(),
# }

# df = pd.DataFrame(newData)
# df.to_csv('CSV-Files/QS_WordLemma_dev1.csv', encoding='utf-8', index=False)


#############################################################
#############################################################

# Extract Min deletion or Insertion to transfrom question to sentence

# data = pd.read_csv("CSV-Files/EditDistance_dev1.csv")
# question_list = data["question"].tolist()
# sentence_list = data["sentence"].tolist()


# def stopword_func(text):
#     stop_words = set(stopwords.words('english'))
#     stop_words.update(('?', 'and', 'I', 'A', 'And', 'So', '.', 'as', 'As', '\'\'', 'could', '[', ']', ',', ')', '\'s', 'By', '(', "''",
#                        'Other', '``', ':', '\'', '#', '\'v', 'The', ';', 'however', 'still',
#                       'the', 'They', 'For', 'also', 'In', 'This', 'When', 'It', 'many', 'Many', 'so', 'cant', 'Yes', 'yes', 'No', 'no',
#                        'These', 'these', 'This', 'Where', 'Which', 'Why', 'How', 'What', 'If', 'Who', 'When'))

#     str1 = ""
#     for ele in text:
#         str1 += ele
#     lemmatizer = WordNetLemmatizer()
#     word_tokens1 = word_tokenize(str1)
#     filtered_sentence = [lemmatizer.lemmatize(
#         word) for word in word_tokens1 if not word in set(stop_words)]

#     return filtered_sentence


# def lcs(Q, S):
#     question = stopword_func(Q)
#     sentence = stopword_func(S)

#     vector = [[0 for i in range(len(sentence) + 1)]
#               for i in range(len(question) + 1)]

#     for Q_index in range(len(question) + 1):
#         for S_index in range(len(sentence) + 1):
#             if (Q_index == 0 or S_index == 0):
#                 vector[Q_index][S_index] = 0

#             elif (question[Q_index - 1] == sentence[S_index - 1]):
#                 vector[Q_index][S_index] = vector[Q_index - 1][S_index - 1] + 1

#             else:
#                 vector[Q_index][S_index] = max(vector[Q_index - 1][S_index],
#                                                vector[Q_index][S_index - 1])

#     # L[m][n] contains length of LCS
#     # for X[0..n-1] and Y[0..m-1]
#     return vector[len(question)][len(sentence)]


# Min_DEL_INS = []
# for index in range(len(sentence_list)):
#     min_QS = lcs(question_list[index], sentence_list[index])
#     Min_DEL_INS.append(
#         min(len(sentence_list[index])-min_QS, len(question_list[index])-min_QS))


# min_del_ins_data = {
#     "question": question_list,
#     "sentence": sentence_list,
#     "MIN_DEL_INS": Min_DEL_INS,
#     "paragraphNo": data["paragraphNo"],
#     "titleNo": data["titleNo"],
# }

# df = pd.DataFrame(min_del_ins_data)
# df.to_csv('CSV-Files/MinDelIns_dev1.csv', encoding='utf-8', index=False)


#############################################################
#############################################################

# Sort MIN_DEL_INS from min to max

ED_data = pd.read_csv("CSV-Files/MinDelIns_dev1.csv")

# with open('CSV-Files/QuestionsList.csv', newline='') as f:
#     reader = csv.reader(f)
#     questions_set = list(reader)

data = pd.read_csv('CSV-Files/QuestionsList.csv')

# questions_set = questions_set[1:]
# questions_set = [question for question in questions_set]
question_dataset = data.loc[data['TitleNo'] == 0].copy()

question_list = []
sentence_list = []
ED_list = []
parag_list = []
title_list = []
question_set = question_dataset['Question'].tolist()

for question in question_set:
    # print(question[3:-3])
    tempData = ED_data.loc[ED_data['question'] == question[3:-3]].copy()

    tempData = ED_data.sort_values(by=['MIN_DEL_INS'],
                                   ascending=True).head(1)

    print(tempData)

    for index in range(len(tempData)):
        question_list.append(tempData.iloc[index]['question'])
        sentence_list.append(tempData.iloc[index]['sentence'])
        ED_list.append(tempData.iloc[index]['MIN_DEL_INS'])
        parag_list.append(tempData.iloc[index]['paragraphNo'])
        title_list.append(tempData.iloc[index]['titleNo'])

    # question_list.append(tempData.iloc[index]['question'])
    # sentence_list.append(tempData.iloc[index]['sentence'])
    # ED_list.append(tempData.iloc[index]['MIN_DEL_INS'])
    # parag_list.append(tempData.iloc[index]['paragraphNo'])
    # title_list.append(tempData.iloc[index]['titleNo'])

newData = {
    "question": question_list,
    "sentence": sentence_list,
    "MIN_DEL_INS": ED_list,
    "paragraphNo": parag_list,
    "titleNo": title_list,
}


df = pd.DataFrame(newData)
df.to_csv('CSV-Files/Sort_MinDelIns_dev1.csv', encoding='utf-8', index=False)


print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
