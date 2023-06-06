import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import datetime
import pandas as pd

from importjson import document_context

# from nltk import pos_tag, word_tokenize, RegexpParser

print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
print("Reading File..")

# document_context = [["Super Bowl 50 was an American football game to determine the champion of the \
#         National Football League (NFL) for the 2015 season.\
#         The American Football Conference (AFC) champion Denver Broncos defeated the \
#         National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. \
#         The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. \
#         As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, \
#         as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals \
#         (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."]]


def semantic_tree(txt):
    stop_words = set(stopwords.words('english'))
    tokenized = sent_tokenize(txt)

    for i in tokenized:

        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words]
        tagged = nltk.pos_tag(wordsList)

        return tagged


tagged_list_result = []
for document in document_context[0]:
    tagged = semantic_tree(document)
    # append multiple NNP
    tagged_list = []
    for tag_index in range(len(tagged)):
        tagged_list.append(tagged[tag_index])
        if tagged[tag_index][1] == "NNP":
            if tag_index != len(tagged):
                index = tag_index
                while tagged[index+1][1] == "NNP":
                    item = tagged_list[-1][0]
                    item = item + " " + tagged[index+1][0]
                    tagged_list.remove(tagged_list[-1])
                    tagged_list.append((item, 'NNP'))
                    index += 1

    tagged_lst = []
    for item1 in tagged_list:
        if tagged_lst == []:
            tagged_lst.append(item1)
        else:
            tagged_keys = [item[0] for item in tagged_lst]
            if item1[0] not in tagged_keys[-1]:
                tagged_lst.append(item1)

    tagged_list_result.append(tagged_lst)

data = {
    'context': document_context[0],
    'tagged': tagged_list_result,
}

df = pd.DataFrame(data)
df.to_csv('CSV-Files/tagged_dev1.csv', encoding='utf-8', index=False)

# print(tagged_lst)
print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
