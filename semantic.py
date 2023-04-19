import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
# from nltk import pos_tag, word_tokenize, RegexpParser

txt = "Sukanya, Rajib and Naba are my good friends. " \
    "Sukanya is getting married next year. " \
    "Marriage is a big step in one’s life." \
    "It is both exciting and frightening. " \
    "But friendship is a sacred bond between people." \
    "It is a special kind of love between us. " \
    "Many of you must have tried searching for a friend "\
    "but never found the right one."


stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(txt)

for i in tokenized:

    wordsList = nltk.word_tokenize(i)
    wordsList = [w for w in wordsList if not w in stop_words]
    tagged = nltk.pos_tag(wordsList)

    print(tagged)
# ///////

# tagged = pos_tag(word_tokenize(txt))
# chunker = RegexpParser("""
#             NP: {<DT>?<JJ>*<NN>}    #Noun Phrases
#             P: {<IN>}               #Prepositions
#             V: {<V.*>}              #Verbs
#             PP: {<P> <NP>}          #Prepostional Phrases
#             VP: {<V> <NP|PP>*}      #Verb Phrases
#                        """)

# output = chunker.parse(tagged)
# print("After Extracting\n",output)


# ////

# blob_object = TextBlob(txt)
# print(blob_object.tags)

# ////
