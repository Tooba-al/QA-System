import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import math
import string
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords


def bigramTFIDF(text):
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word
        for word in text
        if (word not in stop_words) and (word not in string.punctuation)
    ]

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
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word
        for word in text
        if (word not in stop_words) and (word not in string.punctuation)
    ]

    if not filtered_words:
        return 0
    doc = " ".join(filtered_words)
    tfidf = TfidfVectorizer(ngram_range=(3, 3))
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


def calculate_features(question, sentence):
    question_tokens = nltk.word_tokenize(question)
    sentence_tokens = nltk.word_tokenize(sentence)

    matching_words = set(question_tokens).intersection(sentence_tokens)
    mwf = len(matching_words) / len(question_tokens)

    question_bigrams = set(nltk.bigrams(question_tokens))
    sentence_bigrams = set(nltk.bigrams(sentence_tokens))
    bigram_overlap = len(question_bigrams.intersection(sentence_bigrams)) / len(
        question_bigrams
    )

    question_trigrams = set(nltk.trigrams(question_tokens))
    sentence_trigrams = set(nltk.trigrams(sentence_tokens))
    trigram_overlap = len(question_trigrams.intersection(sentence_trigrams)) / len(
        question_trigrams
    )

    vectorizer = TfidfVectorizer()
    span_tfidf = vectorizer.fit_transform([sentence]).toarray()
    question_tfidf = vectorizer.transform([question]).toarray()
    span_tfidf_score = (span_tfidf * question_tfidf.T).sum()
    minkowski_distance = minkowskidistance(question_tokens, sentence_tokens)
    bigram_TFIDF = bigramTFIDF(sentence_tokens)
    trigram_TFIDF = trigramTFIDF(sentence_tokens)

    manhattan_distance = manhattandistance(question_tokens, sentence_tokens)

    jaccard_distance = nltk.distance.jaccard_distance(
        set(question_tokens), set(sentence_tokens)
    )
    euclidean_distance = euclideandistance(question_tokens, sentence_tokens)
    span_TFIDF = spanTFIDF(sentence_tokens)
    return [
        span_tfidf_score,
        mwf,
        bigram_overlap,
        trigram_overlap,
        span_TFIDF,
        bigram_TFIDF,
        trigram_TFIDF,
        minkowski_distance,
        manhattan_distance,
        euclidean_distance,
        jaccard_distance,
    ]


data = pd.read_csv("kNN/KNN_Features_dev.csv", encoding="cp1252")
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data.iloc[:, 2:])
# test_data_norm = scaler.transform(test_data.iloc[:, 2:])

k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_data_norm, train_data["span"])
questionss = []
contextss = []


def process_spans_for_question(testCSV, question):
    rows = testCSV[testCSV["question"] == question]
    spans = rows["span"].tolist()
    new_features = [calculate_features(question, sentence) for sentence in spans]
    new_features_norm = scaler.transform(new_features)
    prob_estimates = knn.predict_proba(new_features_norm)

    # argmax function from NumPy find the index of the sentence with the highest predicted probability. The second column of prob_estimates (indexed by [:, 1]) contains the probability estimates for the positive class (i.e., the sentence with the answer)

    best_sentence_idx = np.argmax(prob_estimates[:, 1])
    best_sentence = spans[best_sentence_idx]
    questionss.append(question)
    contextss.append(best_sentence)


testCSV = pd.read_csv("KNN/TestData_dev1.csv")

# for question in testCSV['question'].unique():
#     process_spans_for_question(testCSV, question)

new_question = "How many total yards did Denver gain?"
new_sentences = [
    "Super Bowl 50 featured numerous records from individuals and teams.",
    "Denver won despite being massively outgained in total yards (315 to 194) and first downs (21 to 11).",
    "Their 194 yards and 11 first downs were both the lowest totals ever by a Super Bowl winning team.",
    "The previous record was 244 yards by the Baltimore Ravens in Super Bowl XXXV.",
    "Only seven other teams had ever gained less than 200 yards in a Super Bowl, and all of them had lost.",
    "The Broncos' seven sacks tied a Super Bowl record set by the Chicago Bears in Super Bowl XX.",
    "Kony Ealy tied a Super Bowl record with three sacks.",
    "Jordan Norwood's 61-yard punt return set a new record, surpassing the old record of 45 yards set by John Taylor in Super Bowl XXIII.",
    "Denver was just 1-of-14 on third down, while Carolina was barely better at 3-of-15.",
    "The two teams' combined third down conversion percentage of 13.8 was a Super Bowl low.",
    "Manning and Newton had quarterback passer ratings of 56.6 and 55.4, respectively, and their added total of 112 is a record lowest aggregate passer rating for a Super Bowl.",
    "Manning became the oldest quarterback ever to win a Super Bowl at age 39, and the first quarterback ever to win a Super Bowl with two different teams, while Gary Kubiak became the first head coach to win a Super Bowl with the same franchise he went to the Super Bowl with as a player.",
]


new_features = [
    calculate_features(new_question, sentence) for sentence in new_sentences
]
print(new_features)
new_features_norm = scaler.transform(new_features)


prob_estimates = knn.predict_proba(new_features_norm)

best_sentence_idx = np.argmax(prob_estimates[:, 1])

best_sentence = new_sentences[best_sentence_idx]
print(best_sentence)
# answer_df = pd.DataFrame({'Question': questionss, 'predicted sentence': contextss})

# answer_df.to_csv('knn_test.csv', index=False)
