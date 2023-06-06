from nltk.stem import WordNetLemmatizer
import datetime
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()


print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))


# dependency parsing tree
DPT = pd.read_csv('CSV-Files/DependencyParsing_dev1.csv')

# question sentence
QS = pd.read_csv('CSV-Files/QS_WordLemma_dev1.csv')

sentence_list = QS['sentence'].tolist()
question_list = QS['question'].tolist()
parag_list = QS['paragraphNo'].tolist()
title_list = QS['titleNo'].tolist()

question_index = []
sentence_index = []
common_word = []
paragraph_no = []
title_no = []
questions_list = []
sentences_list = []


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


def extractAnchors(question, sentence, paragraphNo, titleNo):
    question_words = stopword_func(question)
    sentence_words = stopword_func(sentence)

    for q_word in question_words:
        for s_word in sentence_words:

            qq_word = lemmatizer.lemmatize(q_word)
            ss_word = lemmatizer.lemmatize(s_word)

            if qq_word == ss_word:
                questions_list.append(question)
                sentences_list.append(sentence)
                question_index.append(question_words.index(q_word))
                sentence_index.append(sentence_words.index(s_word))
                common_word.append(q_word)
                paragraph_no.append(paragraphNo)
                title_no.append(titleNo)


for index in range(len(question_list)):
    extractAnchors(question_list[index], sentence_list[index],
                   parag_list[index], title_list[index])

newData = {
    "word": common_word,
    "question": questions_list,
    "sentence": sentences_list,
    "question_index": question_index,
    "sentence_index": sentence_index,
    "paragraphNo": paragraph_no,
    "titleNo": title_no,
}

df = pd.DataFrame(newData)
df.to_csv('CSV-Files/ExtractAnchors_dev1.csv', encoding='utf-8', index=False)


print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
