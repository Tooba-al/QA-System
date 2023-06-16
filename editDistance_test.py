from importjson import document_context, document_question
import pandas as pd
import spacy
import datetime


print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))

nlp = spacy.load("en_core_web_sm")


def edit_distance(question, sentence):
    if len(question) > len(sentence):
        difference = len(question) - len(sentence)
        question[:difference]

    elif len(sentence) > len(question):
        difference = len(sentence) - len(question)
        sentence[:difference]

    else:
        difference = 0

    for word in range(len(question)):
        if word < len(sentence):
            if question[word] != sentence[word]:
                difference += 1
        else:
            difference += 1

    for word in range(len(sentence)):
        if word < len(question):
            if question[word] != sentence[word]:
                difference += 1
        else:
            difference += 1

    return difference


ed_list = []
question_ed = []
sentence_ed = []
parag_no = []
title_no = []
for question in document_question[0]:
    for document in document_context[0]:
        doc = nlp(document)
        # document_list = document.split('. ')
        # print(document_list)
        for sentence in doc.sents:
            parag_no.append(document_context[0].index(document))
            title_no.append(0)
            question_ed.append(question)
            sentence_ed.append(sentence.text)
            # ED = edit_distance(question, sentence.text)
            # ed_list.append(ED)


data = {
    "question": question_ed,
    "sentence": sentence_ed,
    # "edit-distance": ed_list,
    "paragraphNo": parag_no,
    "titleNo": title_no,
}

df = pd.DataFrame(data)
df.to_csv('Features/EditDistance_dev1.csv', encoding='utf-8', index=False)

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
