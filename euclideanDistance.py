import nltk
import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import math
import string
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from csv import writer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist

# def bigramTFIDF(text):
#     stop_words = set(
#         stopwords.words("english")
        
#     )
#     filtered_words =[
#             word
#             for word in text
#             if (word not in stop_words) and (word not in string.punctuation)]
    
#     if not filtered_words:
#         return 0
#     doc = " ".join(filtered_words)
#     tfidf = TfidfVectorizer(ngram_range=(2, 2))
#     tfidf_matrix = tfidf.fit_transform([doc])
#     feature_names = tfidf.get_feature_names_out()

#     if len(feature_names) == 0:
#         return 0
#     tfidf_sum = tfidf_matrix.sum()
#     return tfidf_sum



# def trigramTFIDF(text):
#     stop_words = set(
#         stopwords.words("english")
        
#     )
#     filtered_words =[
#             word
#             for word in text
#             if (word not in stop_words) and (word not in string.punctuation)]
    
#     if not filtered_words:
#         return 0
#     doc = " ".join(filtered_words)
#     tfidf = TfidfVectorizer(ngram_range=(3,3))
#     tfidf_matrix = tfidf.fit_transform([doc])
#     feature_names = tfidf.get_feature_names_out()

#     if len(feature_names) == 0:
#         return 0
#     tfidf_sum = tfidf_matrix.sum()
#     return tfidf_sum 

# def euclideandistance(question_tokens, sentence_tokens):
#     unique_chars = set(question_tokens + sentence_tokens)

#     question_vector = [question_tokens.count(c) for c in unique_chars]
#     span_vector = [sentence_tokens.count(c) for c in unique_chars]

#     squared_distance = sum((x - y) ** 2 for x, y in zip(question_vector, span_vector))
#     return math.sqrt(squared_distance)  
# def minkowskidistance(question_tokens, sentence_tokens, p=1):
#     unique_chars = set(question_tokens + sentence_tokens)

#     question_vector = [question_tokens.count(c) for c in unique_chars]
#     span_vector = [sentence_tokens.count(c) for c in unique_chars]

#     distance = 0
#     for x, y in zip(question_vector, span_vector):
#         distance += abs(x - y) ** p
#     distance **= 1 / p

#     return distance
# def manhattandistance(question_tokens, sentence_tokens):
#         unique_chars = set(question_tokens + sentence_tokens)

#         question_vector = [question_tokens.count(c) for c in unique_chars]
#         span_vector = [sentence_tokens.count(c) for c in unique_chars]

#         manhattan_distance = sum(abs(x - y) for x, y in zip(question_vector, span_vector))

#         return manhattan_distance
# def spanTFIDF(sentence_tokens):
#     result = 0
#     for word in sentence_tokens:
#         tf = 0
#         occurance = 0
#         N = len(sentence_tokens)
#         occurance = sentence_tokens.count(word)

#         if occurance != 0:
#             tf = occurance / N
#         result += tf

#     return result


# def calculate_features(question, sentence):
#     question_tokens = nltk.word_tokenize(question)
#     sentence_tokens = nltk.word_tokenize(sentence)
    
#     matching_words = set(question_tokens).intersection(sentence_tokens)
#     mwf = len(matching_words) / len(question_tokens)
    
#     question_bigrams = set(nltk.bigrams(question_tokens))
#     sentence_bigrams = set(nltk.bigrams(sentence_tokens))
#     bigram_overlap = len(question_bigrams.intersection(sentence_bigrams)) / len(question_bigrams)
    
#     question_trigrams = set(nltk.trigrams(question_tokens))
#     sentence_trigrams = set(nltk.trigrams(sentence_tokens))
#     trigram_overlap = len(question_trigrams.intersection(sentence_trigrams)) / len(question_trigrams)
    
#     vectorizer = TfidfVectorizer()
#     span_tfidf = vectorizer.fit_transform([sentence]).toarray()
#     question_tfidf = vectorizer.transform([question]).toarray()
#     span_tfidf_score = (span_tfidf * question_tfidf.T).sum()
#     minkowski_distance=minkowskidistance(question_tokens, sentence_tokens)
#     bigram_TFIDF=bigramTFIDF(sentence_tokens)
#     trigram_TFIDF=trigramTFIDF(sentence_tokens)

#     manhattan_distance=manhattandistance(question_tokens, sentence_tokens)

#     jaccard_distance = nltk.distance.jaccard_distance(set(question_tokens), set(sentence_tokens))
#     euclidean_distance=euclideandistance(question_tokens, sentence_tokens)
#     span_TFIDF = spanTFIDF(sentence_tokens)
#     # return [span_tfidf_score,mwf,bigram_overlap , trigram_overlap,span_TFIDF,
#     #        bigram_TFIDF, trigram_TFIDF,minkowski_distance, manhattan_distance,euclidean_distance, jaccard_distance]
#     features_data = {
#         "question": question,
#         "span": sentence,
#         "span_TFIDF": span_tfidf_score,
#         "matching_word_frequency": mwf,
#         "bigram_overlap": bigram_overlap,
#         "trigram_overlap": trigram_overlap,
#         "span_word_frequency": span_TFIDF,
#         "bigram_TFIDF": bigram_TFIDF,
#         "trigram_TFIDF": trigram_TFIDF,
#         "minkowski_distance": minkowski_distance,
#         "manhattan_distance": manhattan_distance,
#         "euclidean_distance": euclidean_distance,
#         "jaccard_distance": jaccard_distance,


#     }

#     return features_data


# /////

# trainCSV = pd.read_csv("kNN/KNN_Features_dev.csv",encoding='cp1252')
# print(trainCSV.shape)
# scaler = MinMaxScaler()
# train_data_norm = scaler.fit_transform(trainCSV.iloc[:, 2:])

# testCSV = pd.read_csv("kNN/test_Features_dev.csv",encoding='cp1252')
# print(testCSV.shape)
# test_data_norm = scaler.transform(testCSV.iloc[:, 2:])

# # distances = cdist(test_data_norm, train_data_norm, metric='euclidean')

# # distances = np.zeros((testCSV.shape[0], trainCSV.shape[0]))
# distances_list = []
# for i in range(test_data_norm.shape[0]):
#     for j in range(train_data_norm.shape[0]):
#         distance = np.sqrt(np.sum((test_data_norm[i, :] - train_data_norm[j, :])**2))
#         test_question = testCSV.iloc[i]['question']
#         test_span = testCSV.iloc[i]['span']
#         train_question = trainCSV.iloc[j]['question']
#         train_span = trainCSV.iloc[j]['span']
#         distances_list.append([test_question, test_span, train_question, train_span, distance])
        
# distances_df = pd.DataFrame(distances_list, columns=['test_question', 'test_span', 'train_question', 'train_span', 'distance'])
# distances_df.to_csv('kNN/distances.csv', index=False)

# ////// MIN DISTANCE EACH GROUP
df = pd.read_csv('KNN/distances.csv')
grouped_df = df.groupby(['test_question', 'test_span'])
min_distance_df = grouped_df['distance'].min().reset_index()
min_distance_df.to_csv("KNN/min_distances.csv", index=False)

# /// SORT MIN DISTANCE
df = pd.read_csv("KNN/min_distances.csv")
sorted_df = df.sort_values(by=['test_question', 'distance'])
grouped_df = sorted_df.groupby('test_question', group_keys=False)
sorted_spans_df = grouped_df.apply(lambda x: x.sort_values(by=['distance']))
sorted_spans_df = sorted_spans_df.reset_index(drop=True)
sorted_spans_df.to_csv("KNN/sort_min_distances.csv", index=False)
# ///

# print(distances.shape)
# distance_euc = euclidean_distances(train_data_norm, test_data_norm)
# print(type(distance_euc))
# np.savetxt('euclidean_on_features.csv', distance_euc, delimiter=',')
# distance_cos = cosine_distances(train_data_norm, test_data_norm)
# print(distance_cos)

# FeatureTest = []
# def process_spans_for_question(testCSV, question):
#     rows = testCSV[testCSV['question'] == question]
#     spans = rows['span'].tolist()
    # new_features = [calculate_features(question, sentence) for sentence in spans]

    # FeatureTest.append(new_features)
    
# 




# def main():
#     testCSV = pd.read_csv("KNN/TestData_dev1.csv")
#     # df = pd.read_csv('TrainData_dev1.csv')
#     # data_questions = testCSV["question"]
#     # data_spans = df["span"]
#     dict_datas = []

#     for question in testCSV['question'].unique():
#         # FeatureTest = process_spans_for_question(testCSV, question)
#         rows = testCSV[testCSV['question'] == question]
#         # print (rows)


#         spans = rows['span'].tolist()
#         # print (spans)
#         # this_result = get_features(question, span, answer)
#         new_features = [calculate_features(question, sentence) for sentence in spans]
#         dict_datas.append(new_features)

#     csv_file = "KNN/test_Features_dev.csv"
#     csv_columns = [ 
#         "question",
#         "span",
#         "span_TFIDF",
#         "matching_word_frequency",
#         "bigram_overlap",
#         "trigram_overlap", 
#         "span_word_frequency",
#         "bigram_TFIDF",
#         "trigram_TFIDF",
#         "minkowski_distance",
#         "manhattan_distance",
#         "euclidean_distance",
#         "jaccard_distance",


#     ]
#     try:
#         with open(csv_file, "w", newline='') as features_file:
#             writer = csv.DictWriter(features_file, fieldnames=csv_columns)
#             writer.writeheader()
#             for data in dict_datas:
#                 for question in data:
#                 # Remove any whitespace characters from the data
#                     question = {k: v.strip() if isinstance(v, str) else v for k, v in question.items()}
#                     writer.writerow(question)
#     except IOError:
#         print(Fore.RED + "I/O error")

# main()




# print(FeatureTest)
# answer_df = pd.DataFrame({'Question': questionss, 'predicted sentence': contextss})

# # answer_df.to_csv('knn_test.csv', index=False)
# data_scaled_df = pd.DataFrame(FeatureTest, columns=["span_tfidf_score","mwf","bigram_overlap" , "trigram_overlap","span_TFIDF",
#            "bigram_TFIDF", "trigram_TFIDF","minkowski_distance", "manhattan_distance","euclidean_distance", "jaccard_distance"])

# data_scaled_df['question'] = testCSV['question']
# data_scaled_df['span'] = testCSV['span']

# data_scaled_df.to_csv('test_Features_dev.csv', index=False)

# test_data_norm = scaler.transform(np.array(FeatureTest))
# print(test_data_norm)
# print(FeatureTest)

