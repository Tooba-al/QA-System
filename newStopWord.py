from nltk.corpus import stopwords
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("filtered_tfidf.csv")
# dfq = pd.read_csv("textQ.csv")
dfq = df['question'][0]
print(dfq)


# dfTitleParagraph = df[df.columns[1]]
# dfParagraph = df[df.columns[2]]

# stoplist = stopwords.words('english') + ['though', 'and', 'I', 'A', 'a', 'an', 'An', 'And', 'So', '.', ',', ')', 'By', '(', "''", 'Other', 'The',
#                                          ';', 'however', 'still', 'the', 'They', 'For', 'for', 'also', 'In', 'This', 'When', 'It', 'so', 'Yes', 'yes', 'No', 'no', 'These', 'these', 'This']

# c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2, 3))
# ngrams = c_vec.fit_transform(df[df.columns[0]])

# vocab_ngrams = c_vec.vocabulary_
# ngramsArrya = [((i, j), ngrams[i, j]) for i, j in zip(*ngrams.nonzero())]

# c_vecq = CountVectorizer(ngram_range=(2, 3))

# ngramsq = c_vecq.fit_transform(dfq[dfq.columns[0]])
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

# wordArray2 = list(set(wordArray))

        
            

    
        

# data = {
#     "word": wordArray,
#     "paragraphNo": paragraphNo,
#     "titleNo": titleNo,
#     "count Repeat": countRepeat,

# }

# print(countRepeat)

# df = pd.DataFrame(data)
# df = df.sort_values(by="count Repeat", ascending=False, ignore_index=True)
# # df = df.nlargest(5, "count Repeat")

# df.to_csv('newSlideEdit.csv', encoding='utf-8')
