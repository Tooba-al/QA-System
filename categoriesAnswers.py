import pandas as pd
import json
import spacy
from spacy import displacy
questions1 = []
answers1 = []


# read json file
with open('CSV-Files/devSplit/dev1.json') as f:
    # with open('dev.json') as f:
    data = json.load(f)

# extract contexts1, questions1 and answers1
for i in range(len(data['data'])):
    for j in range(len(data['data'][i]['paragraphs'])):
        for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
            questions1.append(
                data['data'][i]['paragraphs'][j]['qas'][k]['question'])
            answers1.append(
                data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])
# data = {
#     "question": questions1,
#     "answers": answers1

# }

# df = pd.DataFrame(data)
# df.to_csv('extractAnswers.csv', encoding='utf-8')

# //////////////////////////////////////////////////////////////////////////////////

numerical = []
categorical = []
for i in answers1:
    if (i.isdecimal() == True or i.isdigit()== True or i.isnumeric() == True):
        numerical.append(i)
    else:
        categorical.append(i)

# print ("numerical: ",numerical)
# print("categorical: ",categorical)


# ///////////////////////////
cardinals=[]
person =[]
date=[]
loc =[]
money =[]
gpe =[]
org =[]
event =[]
other = []


NER = spacy.load("en_core_web_sm")
for i in answers1:
    text2= NER(i)

    for word in text2.ents:
        if (word.label_ == "CARDINAL"):
            cardinals.append(word.text)
        elif(word.label_ == "PERSON" ):
            person.append(word.text)
        elif(word.label_ == "DATE"):
            date.append(word.text)
        elif(word.label_ == "LOC"):
            loc.append(word.text)
        elif(word.label_ == "MONEY"):
            money.append(word.text)
        elif(word.label_ == "ORG"):
            org.append(word.text)
        elif(word.label_ == "GPE"):
            gpe.append(word.text)
        elif(word.label_ == "EVENT"):
            event.append(word.text)
        else:
            other.append(word.text)

print("other: ",other)


# # df = pd.read_csv('extractAnswers.csv')

# df_numerical = df.select_dtypes(include=['int64'])
# df_categorical = df.select_dtypes(include='object')

# df_numerical_cols = df_numerical.columns.tolist()
# df_categorical_cols = df_categorical.columns.tolist()

# print(df_categorical_cols)