<<<<<<< HEAD
import pandas as pd
import csv
import sqlite3
from nltk.corpus import stopwords
import itertools
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("Filtered_TFIDF/f_dev1.csv")
context = pd.read_csv("ContextList.csv")

question_set = []
question_set.append(data['question'][0])
tempData = data.loc[data['question'].isin(question_set)].copy()

df = pd.DataFrame(tempData)
df.to_csv('filtered_info2.csv', encoding='utf-8', index=False)

# with open('filtered_info.csv', newline='') as f:
#     reader = csv.reader(f)
#     info_set = list(reader)
# info_set = info_set[1:]

# titleno_set=[]
# titleno_set.append(tempData['title_no'][1:])
# titleno_set = [title[4] for title in info_set]
# titleno_set = list(dict.fromkeys(titleno_set))


# title_no_list =[]
# for i in titleno_set:
#     title_no_list.append(int(i))
# paragraphno_set =[]
# paragraphno_set.append(tempData['paragraph_no'][1:])
# paragraphno_set = [paragraph[3] for paragraph in info_set]
# paragraphno_set = list(dict.fromkeys(paragraphno_set))

# parag_no_list =[]
# for i in paragraphno_set:
#     parag_no_list.append(int(i))


# tempData2 = context.loc[context['TitleNo'].isin(title_no_list) & context['ParagraphNo'].isin(parag_no_list) ].copy()
# df = pd.DataFrame(tempData2)
# df.to_csv('filtered_context.csv', encoding='utf-8', index=False)

# ///////////////////////////////////////////////////////////////////////
# df = pd.read_csv("filtered_context.csv")
# dfq = question_set

# dfTitleParagraph = df[df.columns[1]]
# dfParagraph = df[df.columns[2]]

# stoplist = stopwords.words('english') + ['though', 'and', 'I', 'A', 'a', 'an', 'An', 'And', 'So', '.', ',', ')', 'By', '(', "''", 'Other', 'The',
#                                          ';', 'however', 'still', 'the', 'They', 'For', 'for', 'also', 'In', 'This', 'When', 'It', 'so', 'Yes', 'yes', 'No', 'no', 'These', 'these', 'This']

# c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2, 3))
# ngrams = c_vec.fit_transform(df[df.columns[0]])

# vocab_ngrams = c_vec.vocabulary_
# ngramsArrya = [((i, j), ngrams[i, j]) for i, j in zip(*ngrams.nonzero())]

# c_vecq = CountVectorizer(ngram_range=(2, 3))

# ngramsq = c_vecq.fit_transform(dfq)
# vocab_ngramsq = c_vecq.vocabulary_
# # print(vocab_ngramsq)

# ngramsqArrya = [((i, j), ngramsq[i, j]) for i, j in zip(*ngramsq.nonzero())]
# key_vocab_ngramsq = vocab_ngrams.keys()

# wordArray = []
# paragraphNo = []
# titleNo = []
# countRepeat = []

# for item in vocab_ngramsq:
#     if (item in key_vocab_ngramsq):

#         codeP = vocab_ngrams[item]
#         codeQ = vocab_ngramsq[item]
#         tekrar_pragarp = [v for i, v in enumerate(
#             ngramsArrya) if v[0][1] == codeP]
#         tekrar_soal = [v for i, v in enumerate(
#             ngramsqArrya) if v[0][1] == codeQ]
#         Cartesian = itertools.product(tekrar_pragarp, tekrar_soal)
#         for itemCartesian in Cartesian:

#             gav = itemCartesian[0][0][0]
#             gav2 = dfParagraph.values[gav]
#             gav3 = dfTitleParagraph.values[gav]

#             if (item in wordArray):
#                 if (gav2 not in paragraphNo):
#                     if (gav3 not in titleNo):


#                         wordArray.append(item)
#                         paragraphNo.append(gav2)
#                         titleNo.append(gav3)
#                         countRepeat.append(itemCartesian[0][1])

#                     elif (gav3 in titleNo):
#                         wordArray.append(item)
#                         paragraphNo.append(gav2)
#                         titleNo.append(gav3)
#                         countRepeat.append(itemCartesian[0][1])

#                 elif (gav2 in paragraphNo):
#                     if (gav3 not in titleNo):
#                         wordArray.append(item)
#                         paragraphNo.append(gav2)
#                         titleNo.append(gav3)
#                         countRepeat.append(itemCartesian[0][1])
#             else:
#                 wordArray.append(item)
#                 paragraphNo.append(gav2)
#                 titleNo.append(gav3)
#                 countRepeat.append(itemCartesian[0][1])

# data = {
#     "word": wordArray,
#     "paragraphNo": paragraphNo,
#     "titleNo": titleNo,
#     "count Repeat": countRepeat,

# }

# # df = pd.DataFrame(data)
# # df = df.sort_values(by="count Repeat", ascending=False, ignore_index=True)
# # df.to_csv('newSlideEdit.csv', encoding='utf-8')


# with open('newSlideEdit.csv' , 'r') as csvfile:
#     csv_file_reader = csv.reader(csvfile,delimiter=',')
#     next(csv_file_reader,None)
#     index =''
#     word=''
#     paragraphNo=''
#     titleNo=''
#     countRepeat = ''
#     connection=sqlite3.connect('db_csv.db')
#     curosr=connection.cursor()
  
#     # Table_Query = 'CREATE TABLE IF NOT EXISTS newSlide ("index" int, "word" varchar2(25),"paragraphNo" int, "titleNo" int , "countRepeat" int);'
#     # curosr.execute(Table_Query)




#     for row in csv_file_reader:
#         print('row:',row)
#         for i in range(len(row)):
#             index=row[0]
#             word=row[1]
#             paragraphNo=row[2]
#             titleNo=row[3]
#             countRepeat = row[4]
#             # print(countRepeat)

#     # print(word)
#         InsertQuery=f"INSERT INTO newSlide VALUES ('{index}','{word}','{paragraphNo}','{titleNo}','{countRepeat}')"
#         curosr.execute(InsertQuery)
#     data=curosr.execute('''SELECT * 
#                                 FROM newSlide co   
#                                     WHERE ( 
#                                         SELECT COUNT(*) 
#                                             FROM newSlide ci 
#                                                 WHERE  co.word = ci.word 
                                                        
#                                                         AND co.countRepeat < ci.countRepeat ) 
#                                                     < 5''')

#     df = pd.DataFrame(data)
#     df.to_csv('test_max_select.csv', encoding='utf-8',index=False)
#     connection.commit()
#     connection.close()
=======
import pandas as pd
import csv
import sqlite3
from nltk.corpus import stopwords
import itertools
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("Filtered_TFIDF/f_dev1.csv")
context = pd.read_csv("ContextList.csv")

question_set = []
question_set.append(data['question'][0])
tempData = data.loc[data['question'].isin(question_set)].copy()

# df = pd.DataFrame(tempData)
# df.to_csv('filtered_info.csv', encoding='utf-8', index=False)

# with open('filtered_info.csv', newline='') as f:
#     reader = csv.reader(f)
#     info_set = list(reader)
# info_set = info_set[1:]

# titleno_set=[]
# titleno_set.append(tempData['title_no'][1:])
# titleno_set = [title[4] for title in info_set]
# titleno_set = list(dict.fromkeys(titleno_set))


# title_no_list =[]
# for i in titleno_set:
#     title_no_list.append(int(i))
# paragraphno_set =[]
# paragraphno_set.append(tempData['paragraph_no'][1:])
# paragraphno_set = [paragraph[3] for paragraph in info_set]
# paragraphno_set = list(dict.fromkeys(paragraphno_set))

# parag_no_list =[]
# for i in paragraphno_set:
#     parag_no_list.append(int(i))


# tempData2 = context.loc[context['TitleNo'].isin(title_no_list) & context['ParagraphNo'].isin(parag_no_list) ].copy()
# df = pd.DataFrame(tempData2)
# df.to_csv('filtered_context.csv', encoding='utf-8', index=False)

# ///////////////////////////////////////////////////////////////////////
df = pd.read_csv("filtered_context.csv")
dfq = question_set

dfTitleParagraph = df[df.columns[1]]
dfParagraph = df[df.columns[2]]

stoplist = stopwords.words('english') + ['though', 'and', 'I', 'A', 'a', 'an', 'An', 'And', 'So', '.', ',', ')', 'By', '(', "''", 'Other', 'The',
                                         ';', 'however', 'still', 'the', 'They', 'For', 'for', 'also', 'In', 'This', 'When', 'It', 'so', 'Yes', 'yes', 'No', 'no', 'These', 'these', 'This']

c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2, 3))
ngrams = c_vec.fit_transform(df[df.columns[0]])

vocab_ngrams = c_vec.vocabulary_
ngramsArrya = [((i, j), ngrams[i, j]) for i, j in zip(*ngrams.nonzero())]

c_vecq = CountVectorizer(ngram_range=(2, 3))

ngramsq = c_vecq.fit_transform(dfq)
vocab_ngramsq = c_vecq.vocabulary_
# print(vocab_ngramsq)

ngramsqArrya = [((i, j), ngramsq[i, j]) for i, j in zip(*ngramsq.nonzero())]
key_vocab_ngramsq = vocab_ngrams.keys()

wordArray = []
paragraphNo = []
titleNo = []
countRepeat = []

for item in vocab_ngramsq:
    if (item in key_vocab_ngramsq):

        codeP = vocab_ngrams[item]
        codeQ = vocab_ngramsq[item]
        tekrar_pragarp = [v for i, v in enumerate(
            ngramsArrya) if v[0][1] == codeP]
        tekrar_soal = [v for i, v in enumerate(
            ngramsqArrya) if v[0][1] == codeQ]
        Cartesian = itertools.product(tekrar_pragarp, tekrar_soal)
        for itemCartesian in Cartesian:

            gav = itemCartesian[0][0][0]
            gav2 = dfParagraph.values[gav]
            gav3 = dfTitleParagraph.values[gav]

            if (item in wordArray):
                if (gav2 not in paragraphNo):
                    if (gav3 not in titleNo):


                        wordArray.append(item)
                        paragraphNo.append(gav2)
                        titleNo.append(gav3)
                        countRepeat.append(itemCartesian[0][1])

                    elif (gav3 in titleNo):
                        wordArray.append(item)
                        paragraphNo.append(gav2)
                        titleNo.append(gav3)
                        countRepeat.append(itemCartesian[0][1])

                elif (gav2 in paragraphNo):
                    if (gav3 not in titleNo):
                        wordArray.append(item)
                        paragraphNo.append(gav2)
                        titleNo.append(gav3)
                        countRepeat.append(itemCartesian[0][1])
            else:
                wordArray.append(item)
                paragraphNo.append(gav2)
                titleNo.append(gav3)
                countRepeat.append(itemCartesian[0][1])

data = {
    "word": wordArray,
    "paragraphNo": paragraphNo,
    "titleNo": titleNo,
    "count Repeat": countRepeat,

}

# df = pd.DataFrame(data)
# df = df.sort_values(by="count Repeat", ascending=False, ignore_index=True)
# df.to_csv('newSlideEdit.csv', encoding='utf-8')


with open('newSlideEdit.csv' , 'r') as csvfile:
    csv_file_reader = csv.reader(csvfile,delimiter=',')
    next(csv_file_reader,None)
    index =''
    word=''
    paragraphNo=''
    titleNo=''
    countRepeat = ''
    connection=sqlite3.connect('db_csv.db')
    curosr=connection.cursor()
  
    # Table_Query = 'CREATE TABLE IF NOT EXISTS newSlide ("index" int, "word" varchar2(25),"paragraphNo" int, "titleNo" int , "countRepeat" int);'
    # curosr.execute(Table_Query)




    for row in csv_file_reader:
        print('row:',row)
        for i in range(len(row)):
            index=row[0]
            word=row[1]
            paragraphNo=row[2]
            titleNo=row[3]
            countRepeat = row[4]
            # print(countRepeat)

    # print(word)
        InsertQuery=f"INSERT INTO newSlide VALUES ('{index}','{word}','{paragraphNo}','{titleNo}','{countRepeat}')"
        curosr.execute(InsertQuery)
    data=curosr.execute('''SELECT * 
                                FROM newSlide co   
                                    WHERE ( 
                                        SELECT COUNT(*) 
                                            FROM newSlide ci 
                                                WHERE  co.word = ci.word 
                                                        
                                                        AND co.countRepeat < ci.countRepeat ) 
                                                    < 5''')

    df = pd.DataFrame(data)
    df.to_csv('test_max_select.csv', encoding='utf-8',index=False)
    connection.commit()
    connection.close()
>>>>>>> 16c34842653649eafc33b992c5e8812dc5ad7623
