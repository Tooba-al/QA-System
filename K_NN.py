import spacy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import hamming

stop_words = set(stopwords.words('english') + ['though','and','I','A','a','an','An','And','So','.',',',')','By','(',"''",'Other','The',';','however', 'still','the','They','For','for','also','In','This','When','It','so','Yes','yes','No','no','These','these','This'])

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('extractAnswers.csv')

df['answers'] = df['answers'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

def get_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent.label_)
    return entities

df['entities'] = df['answers'].apply(get_entities)
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(df['answers'])

knn = NearestNeighbors(n_neighbors=3)

knn.fit(tfidf)
new_answers = ['Cam Newton.', 'the New England Patriots.', 'the Arizona Cardinals.']

new_tfidf = vectorizer.transform(new_answers)

distances, indices = knn.kneighbors(new_tfidf)

for i in range(len(new_answers)):
    print('New Answer:', new_answers[i])
    for j in indices[i]:
        print('Category:', ', '.join(df['entities'][j]))
    print()

# ///////////////////////////////////////////////////////////////////////////////////////////

# df = pd.read_csv('extractAnswers.csv')

# stop_words = set(stopwords.words('english'))
# df['answers'] = df['answers'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# nlp = spacy.load('en_core_web_sm')

# df['ner'] = df['answers'].apply(lambda x: [(ent.text, ent.label_) for ent in nlp(x).ents])

# new_answers = ['I love playing football', 'I like reading books', 'I enjoy watching movies']

# vectorizer = CountVectorizer(tokenizer=lambda x: [word[0] for word in x])
# X = vectorizer.fit_transform(df['ner'].apply(lambda x: ' '.join(['_'.join(word) for word in x])))

# knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
# knn.fit(X, df['categories'])

# new_X = vectorizer.transform([nlp(answer).ents for answer in new_answers])
# predicted_categories = knn.predict(new_X)

# print(predicted_categories)