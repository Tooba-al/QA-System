import spacy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# nlp = spacy.load("en_core_web_sm")

# df = pd.read_csv("extractAnswers.csv")
# answers = df["answers"].tolist()

# for answer in answers:
#     doc = nlp(answer)
#     for entity in doc.ents:
#         print(entity.text, entity.label_)


# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(answers)
# y = np.array(df["categories"])

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X, y)

# new_answers = ["some new answer", "another new answer"]
# X_new = vectorizer.transform(new_answers)

# y_new = knn.predict(X_new)
# for i, answer in enumerate(new_answers):
#     print(f"Answer: {answer}")
#     print(f"Predicted category: {y_new[i]}")

# ///////////////////////////////////////////////////////////////
stopwords = set(stopwords.words('english'))
stop_words = stopwords.words('english') + ['though','and','I','A','a','an','An','And','So','.',',',')','By','(',"''",'Other','The',';','however', 'still','the','They','For','for','also','In','This','When','It','so','Yes','yes','No','no','These','these','This']

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