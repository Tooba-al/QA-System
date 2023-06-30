# import spacy
import numpy as np
import pandas as pd
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import hamming

# stop_words = set(stopwords.words('english') + ['though','and','I','A','a','an','An','And','So','.',',',')','By','(',"''",'Other','The',';','however', 'still','the','They','For','for','also','In','This','When','It','so','Yes','yes','No','no','These','these','This'])

# nlp = spacy.load('en_core_web_sm')
# df = pd.read_csv('extractAnswers.csv')

# df['answers'] = df['answers'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

# def get_entities(text):
#     doc = nlp(text)
#     entities = []
#     for ent in doc.ents:
#         entities.append(ent.label_)
#     return entities

# df['entities'] = df['answers'].apply(get_entities)
# vectorizer = TfidfVectorizer()

# tfidf = vectorizer.fit_transform(df['answers'])

# knn = NearestNeighbors(n_neighbors=3)

# knn.fit(tfidf)
# new_answers = ['Cam Newton.', 'the New England Patriots.', 'the Arizona Cardinals.']

# new_tfidf = vectorizer.transform(new_answers)

# distances, indices = knn.kneighbors(new_tfidf)

# for i in range(len(new_answers)):
#     print('New Answer:', new_answers[i])
#     for j in indices[i]:
#         print('Category:', ', '.join(df['entities'][j]))
#     print()

# ///////////////////////////////////////////////////////////////////////////////////////////
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(
        stopwords.words("english")
        + [
            "though",
            "and",
            "I",
            "A",
            "a",
            "an",
            "An",
            "And",
            "So",
            ".",
            ",",
            ")",
            "By",
            "(",
            "''",
            "Other",
            "The",
            ";",
            "however",
            "still",
            "the",
            "They",
            "For",
            "for",
            "also",
            "In",
            "This",
            "When",
            "It",
            "so",
            "Yes",
            "yes",
            "No",
            "no",
            "These",
            "these",
            "This",
        ]
    )

stemmer = PorterStemmer()

questions_df = pd.read_csv('CSV-Files/QuestionsList.csv')
contexts_df = pd.read_csv('CSV-Files/ContextList.csv')

questions = questions_df['Question'].tolist()
contexts = contexts_df['Context'].tolist()

tokenized_contexts = []
for context in contexts:
    tokens = context.split()
    tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    tokenized_contexts.append(tokens)

doc_freqs = {}
for context in tokenized_contexts:
    for term in set(context):
        doc_freqs[term] = doc_freqs.get(term, 0) + 1

avg_doc_len = np.mean([len(context) for context in tokenized_contexts])

k1 = 1.2
b = 0.75

N = len(tokenized_contexts)
idf = {}
for term, freq in doc_freqs.items():
    idf[term] = np.log((N - freq + 0.5) / (freq + 0.5))

doc_lens = [len(context) for context in tokenized_contexts]

avg_doc_len = np.mean(doc_lens)

bm25_scores = []
for question in questions:
    tokenized_question = question.split()
    tokenized_question = [stemmer.stem(token) for token in tokenized_question if token.lower() not in stop_words]
    scores = []
    for i, context in enumerate(tokenized_contexts):
        score = 0
        for phrase in tokenized_question:
            if " " in phrase:
                phrase_words = phrase.split()
                if all(word in context for word in phrase_words):
                    idf_term = np.mean([idf.get(word, 0) for word in phrase_words])
                    tf_term = context.count(phrase)
                    doc_len = doc_lens[i]
                    score += idf_term * ((tf_term * (k1 + 1)) / (tf_term + k1 * (1 - b + b * (doc_len / avg_doc_len))))
            else:
                if phrase in doc_freqs:
                    idf_term = idf[phrase]
                    tf_term = context.count(phrase)
                    doc_len = doc_lens[i]
                    score += idf_term * ((tf_term * (k1 + 1)) / (tf_term + k1 * (1 - b + b * (doc_len / avg_doc_len))))
        scores.append(score)
    bm25_scores.append(scores)

top_contexts = []
for scores in bm25_scores:
    top_idxs = np.argsort(scores)[::-1][:3]
    top_contexts.append([contexts[i] for i in top_idxs])

answer_df = pd.DataFrame({'Question': questions, 'Answer': top_contexts})

answer_df.to_csv('BM25_Ranking.csv', index=False)

