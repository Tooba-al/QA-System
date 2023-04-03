from nltk.stem import WordNetLemmatizer
from stopword import filtered_sentence
from stopword import filtered_sentence2
from stopword import filtered_sentence3

lemmatizer = WordNetLemmatizer()
for i in filtered_sentence:
    if(i==lemmatizer.lemmatize(i)):
        continue
    else:
        print(i, ':' ,lemmatizer.lemmatize(i))

for i in filtered_sentence2:
    if(i==lemmatizer.lemmatize(i)):
        continue
    else:
        print(i, ':' ,lemmatizer.lemmatize(i))
        
for i in filtered_sentence3:
    if(i==lemmatizer.lemmatize(i)):
        continue
    else:
        print(i, ':' ,lemmatizer.lemmatize(i))
    
# print("rocks :", lemmatizer.lemmatize("rocks"))
# print("corpora :", lemmatizer.lemmatize("corpora"))
# print("better :", lemmatizer.lemmatize("better", pos ="a"))