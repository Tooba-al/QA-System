import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import math
import string

from nltk.corpus import stopwords

def bigramTFIDF(text):
    stop_words = set(
        stopwords.words("english")
        
    )
    filtered_words =[
            word
            for word in text
            if (word not in stop_words) and (word not in string.punctuation)]
    
    if not filtered_words:
        return 0
    doc = " ".join(filtered_words)
    tfidf = TfidfVectorizer(ngram_range=(2, 2))
    tfidf_matrix = tfidf.fit_transform([doc])
    feature_names = tfidf.get_feature_names_out()

    if len(feature_names) == 0:
        return 0
    tfidf_sum = tfidf_matrix.sum()
    return tfidf_sum



def trigramTFIDF(text):
    stop_words = set(
        stopwords.words("english")
        
    )
    filtered_words =[
            word
            for word in text
            if (word not in stop_words) and (word not in string.punctuation)]
    
    if not filtered_words:
        return 0
    doc = " ".join(filtered_words)
    tfidf = TfidfVectorizer(ngram_range=(3,3))
    tfidf_matrix = tfidf.fit_transform([doc])
    feature_names = tfidf.get_feature_names_out()

    if len(feature_names) == 0:
        return 0
    tfidf_sum = tfidf_matrix.sum()
    return tfidf_sum 

def euclideandistance(question_tokens, sentence_tokens):
    unique_chars = set(question_tokens + sentence_tokens)

    question_vector = [question_tokens.count(c) for c in unique_chars]
    span_vector = [sentence_tokens.count(c) for c in unique_chars]

    squared_distance = sum((x - y) ** 2 for x, y in zip(question_vector, span_vector))
    return math.sqrt(squared_distance)  
def minkowskidistance(question_tokens, sentence_tokens, p=1):
    unique_chars = set(question_tokens + sentence_tokens)

    question_vector = [question_tokens.count(c) for c in unique_chars]
    span_vector = [sentence_tokens.count(c) for c in unique_chars]

    distance = 0
    for x, y in zip(question_vector, span_vector):
        distance += abs(x - y) ** p
    distance **= 1 / p

    return distance
def manhattandistance(question_tokens, sentence_tokens):
        unique_chars = set(question_tokens + sentence_tokens)

        question_vector = [question_tokens.count(c) for c in unique_chars]
        span_vector = [sentence_tokens.count(c) for c in unique_chars]

        manhattan_distance = sum(abs(x - y) for x, y in zip(question_vector, span_vector))

        return manhattan_distance
def spanTFIDF(sentence_tokens):
    result = 0
    for word in sentence_tokens:
        tf = 0
        occurance = 0
        N = len(sentence_tokens)
        occurance = sentence_tokens.count(word)

        if occurance != 0:
            tf = occurance / N
        result += tf

    return result

# define a function to calculate the features for a given question and sentence
def calculate_features(question, sentence):
    # calculate syntactic divergent
    question_tokens = nltk.word_tokenize(question)
    sentence_tokens = nltk.word_tokenize(sentence)
    
    # calculate matching word frequency
    matching_words = set(question_tokens).intersection(sentence_tokens)
    mwf = len(matching_words) / len(question_tokens)
    
    # calculate bigram overlap
    question_bigrams = set(nltk.bigrams(question_tokens))
    sentence_bigrams = set(nltk.bigrams(sentence_tokens))
    bigram_overlap = len(question_bigrams.intersection(sentence_bigrams)) / len(question_bigrams)
    
    # calculate trigram overlap
    question_trigrams = set(nltk.trigrams(question_tokens))
    sentence_trigrams = set(nltk.trigrams(sentence_tokens))
    trigram_overlap = len(question_trigrams.intersection(sentence_trigrams)) / len(question_trigrams)
    
    # calculate span tf-idf
    vectorizer = TfidfVectorizer()
    span_tfidf = vectorizer.fit_transform([sentence]).toarray()
    question_tfidf = vectorizer.transform([question]).toarray()
    span_tfidf_score = (span_tfidf * question_tfidf.T).sum()
    minkowski_distance=minkowskidistance(question_tokens, sentence_tokens)
    bigram_TFIDF=bigramTFIDF(sentence_tokens)
    trigram_TFIDF=trigramTFIDF(sentence_tokens)

    manhattan_distance=manhattandistance(question_tokens, sentence_tokens)
    # calculate other distance-based features
    # hamming_distance=hammingdistance(question_tokens, sentence_tokens)
    jaccard_distance = nltk.distance.jaccard_distance(set(question_tokens), set(sentence_tokens))
    euclidean_distance=euclideandistance(question_tokens, sentence_tokens)
    span_TFIDF = spanTFIDF(sentence_tokens)
    # return the calculated features as a dictionary
    return [span_tfidf_score,mwf,bigram_overlap , trigram_overlap,span_TFIDF,
           bigram_TFIDF, trigram_TFIDF,minkowski_distance, manhattan_distance,euclidean_distance, jaccard_distance]


# calculate the features for the question and each candidate sentence
question = "What is the capital of France?"
candidate_sentences = ["Paris is the capital of France.",
                       "London is the capital of England.",
                       "Berlin is the capital of Germany."]
features = []
for sentence in candidate_sentences:
    features.append(calculate_features(question, sentence))

# scale the features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# load the CSV file into a pandas DataFrame containing the features and labels
df = pd.read_csv("kNN/KNN_Features_dev.csv",encoding='cp1252')

# separate the features and labels
X = df.drop(["span","question"], axis=1)
y = df["span"]

# scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train a KNN classifier using the entire dataset
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_scaled, y)

# for each set of candidate features, calculate the predicted sentence that contains the answer
predicted_sentences = []
for f in features_scaled:
    predicted_sentences.append(knn.predict(f.reshape(1, -1))[0])

# print the predicted sentence for each set of candidate features
for i, sentence in enumerate(candidate_sentences):
    print(f"Question: {question}")
    print(f"Candidate sentence: {sentence}")
    print(f"Predicted sentence: {predicted_sentences[i]}")
    print()  # add a blank line for readability