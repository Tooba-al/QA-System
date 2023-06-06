import spacy
from spacy import displacy
import pandas as pd
import numpy as np
from importjson import document_context
import datetime

print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))

nlp = spacy.load("en_core_web_sm")
sentence = "The quick brown fox jumping over the lazy dog"

dependency = []
head = []
dependent = []
index_list = []
title_index = []
sentence_list = []


for paragraph in document_context[0]:
    doc = nlp(paragraph)
    for sentence in doc.sents:
        # print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")

        for token in sentence:
            sentence_list.append(sentence.text)
            index = document_context[0].index(paragraph)
            title_index.append(0)
            index_list.append(index)
            dependency.append(token.dep_)
            head.append(token.head.text)
            dependent.append(token.text)
            # print("{:<15} {:^10} {:>15}".format(
            #     str(token.head.text), str(token.dep_), str(token.text)))

        # displacy.render(doc, style='dep')


data = {
    'sentence': sentence_list,
    'dependency': dependency,
    'head': head,
    'dependent': dependent,
    'paragraphNo': index_list,
    'titleNo': title_index,
}

df = pd.DataFrame(data)
df.to_csv('CSV-Files/DependencyParsing_dev11.csv',
          encoding='utf-8', index=False)

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
