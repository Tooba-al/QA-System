from nltk.corpus import stopwords
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("ContextList.csv")
dfq = pd.read_csv("QuestionsList.csv")

dfqTitleParagraph = dfq[dfq.columns[1]]
dfqParagraph = dfq[dfq.columns[2]]

dfTitleParagraph = df[df.columns[1]]
dfParagraph = df[df.columns[2]]
# print(dfParagraph.values[0])


stoplist = stopwords.words('english') + ['though','and','I','A','a','an','An','And','So','.',',',')','By','(',"''",'Other','The',';','however', 'still','the','They','For','for','also','In','This','When','It','so','Yes','yes','No','no','These','these','This']


c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
ngrams = c_vec.fit_transform(df[df.columns[0]])

# print(df[df.columns[1]].values[2061])

vocab_ngrams= c_vec.vocabulary_
ngramsArrya=[((i, j), ngrams[i,j]) for i, j in zip(*ngrams.nonzero())]

ngramsq = c_vec.fit_transform(dfq[dfq.columns[0]])
vocab_ngramsq= c_vec.vocabulary_
# print(ngramsArrya)
ngramsqArrya=[((i, j), ngramsq[i,j]) for i, j in zip(*ngramsq.nonzero())]
key_vocab_ngramsq=vocab_ngrams.keys()
wordArray=[]
questionNo=[]
paragraphNo=[]
titleNo=[]
countRepeat=[]

questionBelongsToParagraphNo=[]
questionBelongsToTitleNo=[]

for item in vocab_ngramsq:
    if( item in key_vocab_ngramsq):
    
        codeP=vocab_ngrams[item]
        codeQ=vocab_ngramsq[item]
        tekrar_pragarp=[v for i, v in enumerate(ngramsArrya) if v[0][1] == codeP]
        tekrar_soal=[v for i, v in enumerate(ngramsqArrya) if v[0][1] == codeQ]
        Cartesian=itertools.product(tekrar_pragarp,tekrar_soal)
        for itemCartesian in Cartesian:
            # if (item not in wordArray):
                wordArray.append(item)
                questionNo.append(itemCartesian[1][0][0])

                khar = itemCartesian[1][0][0]
                questionBelongsToTitleNo.append(dfqTitleParagraph.values[khar])
                questionBelongsToParagraphNo.append(dfqParagraph.values[khar])

                gav = itemCartesian[0][0][0]
                
                paragraphNo.append(dfParagraph.values[gav])
                titleNo.append(dfTitleParagraph.values[gav])
                countRepeat.append(itemCartesian[0][1])
            

data={
    "word":wordArray,
    "questionNo":questionNo,
    "paragraphNo":paragraphNo,
    "titleNo":titleNo,
    "count Repeat":countRepeat,
    "question belongs to paragraph" : questionBelongsToParagraphNo,
    "question belongs to paragraph in title" : questionBelongsToTitleNo

}
# print(wordArray)
df = pd.DataFrame(data)
compression_opts = dict(method='zip',archive_name='result.csv')  
df.to_csv('result.zip', index=False,compression=compression_opts)  



# # # count_values = ngramsq.toarray().sum(axis=0)
# # vocab1 = c_vec.vocabulary_
# # # df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
# # #             ).rename(columns={0: 'frequency', 1:'bigram/trigram'})



