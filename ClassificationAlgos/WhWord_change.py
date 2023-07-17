import csv
import pandas as pd
from colorama import Fore

inputCSV = "ClassificationAlgos/Features_dev1.csv"
outputCSV = "ClassificationAlgos/Features_dev1_1.csv"

df = pd.read_csv("ClassificationAlgos/Features_dev1.csv", encoding="utf8")
question_length = df["question_length"].tolist()
span_length = df["span_length"].tolist()
# span_POS_tags = df["span_POS_tags"].tolist()
# consistant_labels = df["consistant_labels"].tolist()
edit_distance = df["edit_distance"].tolist()
jaccard_distance = df["jaccard_distance"].tolist()
hamming_distance = df["hamming_distance"].tolist()
euclidean_distance = df["euclidean_distance"].tolist()
manhattan_distance = df["manhattan_distance"].tolist()
minkowski_distance = df["minkowski_distance"].tolist()
trigram_TFIDF = df["trigram_TFIDF"].tolist()
bigram_TFIDF = df["bigram_TFIDF"].tolist()
span_word_frequency = df["span_word_frequency"].tolist()
trigram_overlap = df["trigram_overlap"].tolist()
bigram_overlap = df["bigram_overlap"].tolist()
matching_word_frequency = df["matching_word_frequency"].tolist()
span_TFIDF = df["span_TFIDF"].tolist()
root_matching = df["root_matching"].tolist()
syntactic_divergence = df["syntactic_divergence"].tolist()
answer = df["answer"].tolist()
span = df["span"].tolist()
question = df["question"].tolist()
titleNo = df["titleNo"].tolist()
paragNo = df["paragNo"].tolist()
wh_word = []
wh_words = [
    "which",
    "where",
    "what",
    "when",
    "who",
    "whom",
    "whose",
    "why",
    "how many",
    "how much",
    "how",
]
wh_words_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for q in question:
    flag = 0
    for index in range(len(wh_words)):
        q = q.lower()
        if q.count(wh_words[index]) != 0:
            flag = 1
            wh_word.append(wh_words_index[index])
            break
    if flag != 1:
        wh_word.append(-1)


# print(len(question_length))
# print(len(span_length))
# print(len(edit_distance))
# print(len(jaccard_distance))
# print(len(hamming_distance))
# print(len(euclidean_distance))
# print(len(manhattan_distance))
# print(len(minkowski_distance))
# print(len(trigram_TFIDF))
# print(len(bigram_TFIDF))
# print(len(span_word_frequency))
# print(len(trigram_overlap))
# print(len(bigram_overlap))
# print(len(matching_word_frequency))
# print(len(span_TFIDF))
# print(len(root_matching))
# print(len(syntactic_divergence))
# print(len(wh_word))
# print(len(answer))
# print(len(span))
# print(len(question))
# print(len(titleNo))
# print(len(paragNo))


features_data = {
    "paragNo": titleNo,
    "titleNo": paragNo,
    "question": question,
    "span": span,
    "answer": answer,
    "wh_word": wh_word,
    "syntactic_divergence": syntactic_divergence,
    "root_matching": root_matching,
    "bigram_overlap": bigram_overlap,
    "trigram_overlap": trigram_overlap,
    "matching_word_frequency": matching_word_frequency,
    "span_word_frequency": span_word_frequency,
    "span_TFIDF": span_TFIDF,
    "bigram_TFIDF": bigram_TFIDF,
    "trigram_TFIDF": trigram_TFIDF,
    "minkowski_distance": minkowski_distance,
    "manhattan_distance": manhattan_distance,
    "euclidean_distance": euclidean_distance,
    "hamming_distance": hamming_distance,
    "jaccard_distance": jaccard_distance,
    "edit_distance": edit_distance,
    "span_length": span_length,
    "question_length": question_length,
}

df = pd.DataFrame(features_data)
df.to_csv("ClassificationAlgos/Features_dev1_1.csv", index=False)
