import pandas as pd
import csv

data_cosinee = pd.read_csv("Filtered_TFIDF/f_dev1.csv")
data_questions = pd.read_csv("ContextList.csv")
p =[]
t=[]
for i in data_cosinee:
    for j in data_questions:
        question_set = []
        question_set.append(data_questions['question'][j])
        temp = data_cosinee.loc[data_cosinee['question'].isin(question_set)]
        if temp != None:
            t.append(temp['TitleNo'])
            p.append(temp['ParagraphNo'])

