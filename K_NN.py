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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
train_data = pd.read_csv("KNN/KNN_Features.csv",encoding='cp1252')
test_data = pd.read_csv("KNN/TestData_dev1.csv")

# Preprocess the data by separating features and labels
train_data = train_data.drop(["answer", "span", "question","syntatic_divergence","consistant_labels","span_POS_tags"], axis=1)

# Preprocess the data by separating features and labels
X_train = train_data
y_train = test_data["question"]

X_test = test_data.drop(["paragraphNo","titleNo"],axis=1)
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Train the KNN classifier using the training set
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_train, y_train_train)

# Evaluate the performance of the trained KNN model on the validation set
val_preds = knn.predict(X_train_val)
accuracy = (val_preds == y_train_val).mean()
print("Validation accuracy:", accuracy)

# Use the trained KNN model to predict answers on the test set
test_preds = []
for i in range(len(X_test_scaled)):
    neighbors = knn.kneighbors([X_test_scaled[i]], n_neighbors=5, return_distance=False)
    majority_class = y_train[neighbors].mode()[0]
    test_preds.append(majority_class)

# Output the predicted answers to a new CSV file
test_data["predicted_answer"] = test_preds
test_data.to_csv("test_with_predictions.csv", index=False)