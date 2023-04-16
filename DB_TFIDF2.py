import pandas as pd
import csv

import datetime

###############################################################################

# print("Start Time: = %s:%s:%s" % (e.hour, e.minute, e.second))
# data = pd.read_csv("outputs/output_dev1.csv")

# questions = data['question']
# questions = questions.drop_duplicates()
# qdf = pd.DataFrame(questions)
# qdf.to_csv('questions1.csv', encoding='utf-8', index=False)

###############################################################################

print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
data = pd.read_csv("outputs/output_dev_1.csv")

# delete duplicate questions
with open('questions1.csv', newline='') as f:
    reader = csv.reader(f)
    questions_set = list(reader)

questions_set = questions_set[1:]
questions_set = [question[0] for question in questions_set]

words = data['word'].to_list()
TF_IDFs = data['TF_IDF'].to_list()
parag_nos = data['paragraph_no'].to_list()
title_nos = data['title_no'].to_list()


question_list = []
word_list = []
TF_IDF_list = []
parag_no_list = []
title_no_list = []
dict_list = []

for item in questions_set:
    tempData = data.loc[data['question'] == item].copy()
    tempData['TF_IDF'].mul(1000)
    tempData['TF_IDF'] = tempData['TF_IDF'].astype(int)
    tempData['TF_IDF'] = tempData['TF_IDF'].astype(str)
    words_2 = list(dict.fromkeys(tempData['word'].values))

    for word in words_2:
        datatTempWord = tempData[tempData['word'] == word]
        df = datatTempWord.sort_values(by=['TF_IDF'],
                                       ascending=False).head(5)
        index = df.index
        filter_data = data[data.index.isin(index)].reset_index(drop=True)

        for row_index in range(len(filter_data)):
            question_list.append(filter_data.iloc[row_index]['question'])
            word_list.append(filter_data.iloc[row_index]['word'])
            TF_IDF_list.append(filter_data.iloc[row_index]['TF_IDF'])
            parag_no_list.append(filter_data.iloc[row_index]['paragraph_no'])
            title_no_list.append(filter_data.iloc[row_index]['title_no'])

df = {
    'question': question_list,
    'word': word_list,
    'TF_IDF': TF_IDF_list,
    'paragraph_no': parag_no_list,
    'title_no': title_no_list,
}

df = pd.DataFrame(df)
df.to_csv('filtered_tfidf_2.csv', encoding='utf-8', index=False)

###############################################################################

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
