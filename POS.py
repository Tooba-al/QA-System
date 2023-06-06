from nltk import pos_tag
from nltk import RegexpParser, word_tokenize
from importjson import document_context, document_question
import pandas as pd
import spacy
import datetime


print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))

nlp = spacy.load("en_core_web_sm")

sents_list = []
pos_list = []
parag_list = []
title_list = []

for question in document_question[0]:
    for document in document_context[0]:
        doc = nlp(document)

        for sentence in doc.sents:
            tokens = word_tokenize(sentence.text)
            pos = pos_tag(tokens, tagset="universal")
            # print(tokens_tag)
            # patterns = """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
            # chunker = RegexpParser(patterns)
            # pos = chunker.parse(tokens_tag)

            sents_list.append(sentence.text)
            pos_list.append(pos)
            parag_list.append(document_context[0].index(document))
            title_list.append(0)


data = {
    "sentence": sents_list,
    "POS": pos_list,
    "paragraphNo": parag_list,
    "titleNo": title_list,
}

df = pd.DataFrame(data)
df.to_csv('CSV-Files/POS_dev1.csv', encoding='utf-8', index=False)

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
