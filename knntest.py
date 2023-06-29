import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import math

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


    manhattan_distance=manhattandistance(question_tokens, sentence_tokens)
    # calculate other distance-based features
    # hamming_distance=hammingdistance(question_tokens, sentence_tokens)
    jaccard_distance = nltk.distance.jaccard_distance(set(question_tokens), set(sentence_tokens))
    euclidean_distance=euclideandistance(question_tokens, sentence_tokens)   
    # return the calculated features as a dictionary
    return [mwf, bigram_overlap, trigram_overlap, span_tfidf_score,
            minkowski_distance, manhattan_distance, jaccard_distance, euclidean_distance]


# calculate the features for the question and each candidate sentence
question = "What is the capital of France?"
candidate_sentences = ["Paris is the capital of France.",
                       "London is the capital of England.",
                       "Berlin is the capital of Germany."]
features = []
for sentence in candidate_sentences:
    features.append(calculate_features(question, sentence))
print("feat: ",features)
# scale the features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# load the CSV file into a pandas DataFrame containing the features and labels
df = pd.read_csv("my_csv_file.csv")

# separate the features and labels
X = df.drop(["sentence"], axis=1)
y = df["sentence"]

# scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train a KNN classifier using the entire dataset
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
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