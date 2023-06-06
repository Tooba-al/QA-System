import pandas as pd
import csv

import datetime

###############################################################################

print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
data = pd.read_csv("CSV-Files/TFIDF_CS_dev2.csv")

questions = data['question']
questions = questions.drop_duplicates()
qdf = pd.DataFrame(questions)
qdf.to_csv('CSV-Files/questions2.csv', encoding='utf-8', index=False)

###############################################################################

data = pd.read_csv("CSV-Files/TFIDF_CS_dev2.csv")

# delete duplicate questions
with open('CSV-Files/questions2.csv', newline='') as f:
    reader = csv.reader(f)
    questions_set = list(reader)

questions_set = questions_set[1:]
questions_set = [question[0] for question in questions_set]

questions = data['question'].to_list()
TF_IDFs = data['TFIDF_CosineSim'].to_list()
doc_nos = data['document_no'].to_list()
title_nos = data['title_no'].to_list()


question_list = []
TF_IDF_list = []
doc_no_list = []
title_no_list = []
dict_list = []

for question in questions_set:
    tempData = data.loc[data['question'] == question].copy()
    tempData['TFIDF_CosineSim'] = tempData['TFIDF_CosineSim'].mul(10**6)
    tempData['TFIDF_CosineSim'] = tempData['TFIDF_CosineSim'].astype(int)

    tempData = tempData.sort_values(by=['TFIDF_CosineSim'],
                                    ascending=False).head(5)
    tempData['TFIDF_CosineSim'] = tempData['TFIDF_CosineSim'].astype(
        float)
    tempData['TFIDF_CosineSim'] = tempData['TFIDF_CosineSim'].mul(
        10**(-6))

    for row_index in range(len(tempData)):

        question_list.append(tempData.iloc[row_index]['question'])
        TF_IDF_list.append(tempData.iloc[row_index]['TFIDF_CosineSim'])
        doc_no_list.append(tempData.iloc[row_index]['document_no'])
        title_no_list.append(tempData.iloc[row_index]['title_no'])

df = {
    'question': question_list,
    'TFIDF_CosineSim': TF_IDF_list,
    'paragraph_no': doc_no_list,
    'title_no': title_no_list,
}

df = pd.DataFrame(df)
df.to_csv('CSV-Files/MAX_TFIDF_CS_dev2.csv', encoding='utf-8', index=False)

###############################################################################

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
