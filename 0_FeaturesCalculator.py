import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import nltk
from nltk import pos_tag
from nltk import word_tokenize
import stanza
import networkx as nx
from scipy.spatial import distance
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
<<<<<<< HEAD
import string
=======
from csv import writer
import datetime
from colorama import Fore
from nltk.tokenize import TreebankWordTokenizer
import time

start_time = time.time()
print(
    Fore.RED
    + "Start Time: = %s:%s:%s"
    % (
        datetime.datetime.now().hour,
        datetime.datetime.now().minute,
        datetime.datetime.now().second,
    )
)

>>>>>>> 1f2ed0f5dbd1b216bf500b4ddc5feb2b33e44af5

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

print(Fore.RED + "Dataset : CSV-Files/devSplit/dev1.json\n")
with open("CSV-Files/devSplit/dev1.json") as f:
    data = json.load(f)

##############################################################################################################
##############################################################################################################
stop_words = set(stopwords.words("english"))
stop_words.update(
        (
            "?",
            "and",
            "I",
            "A",
            "And",
            "So",
            ".",
            "as",
            "As",
            "''",
            "could",
            "[",
            "]",
            ",",
            ")",
            "'s",
            "By",
            "(",
            "''",
            "Other",
            "``",
            ":",
            "'",
            "#",
            "'v",
            "The",
            ";",
            "however",
            "still",
            "the",
            "They",
            "For",
            "also",
            "In",
            "This",
            "When",
            "It",
            "many",
            "Many",
            "so",
            "cant",
            "Yes",
            "yes",
            "No",
            "no",
            "These",
            "these",
            "This",
            "Where",
            "Which",
            "Why",
            "How",
            "What",
            "If",
            "Who",
            "When",
        )
    )
def remove_stopwords_punctuation(text):

    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text.lower())

    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# ////////////////////


def stopword_func(text):
    stop_words = set(stopwords.words("english"))
    stop_words.update(
        (
            "?",
            "and",
            "I",
            "A",
            "And",
            "So",
            ".",
            "as",
            "As",
            "''",
            "could",
            "[",
            "]",
            ",",
            ")",
            "'s",
            "By",
            "(",
            "''",
            "Other",
            "``",
            ":",
            "'",
            "#",
            "'v",
            "The",
            ";",
            "however",
            "still",
            "the",
            "They",
            "For",
            "also",
            "In",
            "This",
            "When",
            "It",
            "many",
            "Many",
            "so",
            "cant",
            "Yes",
            "yes",
            "No",
            "no",
            "These",
            "these",
            "This",
            "Where",
            "Which",
            "Why",
            "How",
            "What",
            "If",
            "Who",
            "When",
        )
    )
    str1 = ""
    for ele in text:
        str1 += ele
    lemmatizer = WordNetLemmatizer()
    word_tokens1 = word_tokenize(str1)
    filtered_span = [
        lemmatizer.lemmatize(word)
        for word in word_tokens1
        if not word in set(stop_words)
    ]

    return filtered_span


def dependency_parser(span):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(span)
    token_positions = []

    # to see all the children token
    for token in doc:
        token_positions.append(str(token) + "-" + str(token.i))

    return token_positions


def find_answer_sentence(paragraph_no, answer):
    contexts = data["data"][0]["paragraphs"][paragraph_no]["context"]
    sentences = nltk.sent_tokenize(contexts)
    result = ""
    for sentence in sentences:
        if answer in sentence:
            result = sentence

    return result


def Q_shortest_dependency_path(anchor, wh_word, text):
    tokens_position = dependency_parser(text)

    document = nlp(text)

    anchor = anchor.lower()
    wh_word = wh_word.lower()

    edges = []
    for token in document:
        for child in token.children:
            edges.append(
                (
                    "{0}-{1}".format(token.lower_, token.i),
                    "{0}-{1}".format(child.lower_, child.i),
                )
            )

    anchor_token = ""
    wh_token = ""
    for token in tokens_position:
        token = token.lower()
        if anchor in token:
            anchor_token = token
        elif wh_word in token:
            wh_token = token

    graph = nx.Graph(edges)
    try:
        path_length = nx.shortest_path_length(
            graph, source=wh_token, target=anchor_token
        )
        path = nx.shortest_path(graph, source=wh_token, target=anchor_token)
    except:
        path_length = 999999
        path = []

    return (path_length, path)


def A_shortest_dependency_path(anchor, answer, text):
    tokens_position = dependency_parser(text)

    document = nlp(text)

    anchor = anchor.lower()
    answer = answer.lower()

    if answer:
        if answer.count('"') != 0:
            answer = answer.replace('"', "")

        if len(answer.split()) > 1:
            answer = answer.split()[0]

        edges = []
        for token in document:
            for child in token.children:
                edges.append(
                    (
                        "{0}-{1}".format(token.lower_, token.i),
                        "{0}-{1}".format(child.lower_, child.i),
                    )
                )

        anchor_token = ""
        answer_token = ""
        for token in tokens_position:
            token = token.lower()
            if anchor in token:
                anchor_token = token
            elif answer in token:
                answer_token = token

        graph = nx.Graph(edges)

        try:
            path_length = nx.shortest_path_length(
                graph, source=anchor_token, target=answer_token
            )
            path = nx.shortest_path(graph, source=anchor_token, target=answer_token)

        except:
            path_length = 999999
            path = []

        return (path_length, path)


def edit_distance(Q_SDP, A_SDP):
    ed = 0
    if Q_SDP[0] > A_SDP[0]:
        ed = Q_SDP[0] - A_SDP[0]
    else:
        ed = A_SDP[0] - Q_SDP[0]

    return ed


def get_syntatic_div(question, span, answer):
    paragraph_no = 0

    tokenizer = TreebankWordTokenizer()
    question_tokens = tokenizer.tokenize(question)
    span_tokens = tokenizer.tokenize(span)

    anchors = []
    for word in question_tokens:
        if span_tokens.count(word) != 0:
            anchors.append(word)

    answer_SDP = []
    question_SDP = []
    anchor_list = []

    [anchor_list.append(item) for item in anchors if item not in anchor_list]
    for anchor in anchor_list:
        question_SDP.append(
            Q_shortest_dependency_path(anchor, question.split()[0], question)
        )
        answer_SDP.append(A_shortest_dependency_path(anchor, answer, span))

    ED_list = []
    for index in range(len(question_SDP)):
        q_SDP = question_SDP[index]
        a_SDP = answer_SDP[index]

        ED = edit_distance(a_SDP, q_SDP)
        ED_list.append(ED)

    if ED_list != []:
        min_ED = min(ED_list)
        return (
            min_ED,
            (answer_SDP[ED_list.index(min_ED)], question_SDP[ED_list.index(min_ED)]),
        )

    else:
        min_ED = -1

        return (
            min_ED,
            ([], []),
        )


def get_root_matching(question, span):
    q_doc = nlp(question)
    s_doc = nlp(span)
    question_dep = ""
    span_dep = ""

    for token in q_doc:
        if token.dep_ == "ROOT":
            question_dep = token

    for token in s_doc:
        if token.dep_ == "ROOT":
            span_dep = token

    if question_dep == span_dep:
        return True
    return False


def get_span_TFIDF(text):
    text = text.split()
    result = 0
    for word in text:
        tf = 0
        occurance = 0
        N = len(text)
        occurance = text.count(word)

        if occurance != 0:
            tf = occurance / N
        result += tf

    return result


def get_matching_word_frequency(question, span):
    question = stopword_func(question)
    span = stopword_func(span)

    result = 0
    for word in question:
        tf = 0
        occurance = span.count(word)
        if occurance > 0:
            tf = occurance / len(span)
            result += tf

    return result


def get_bigram_overlap(question, span):
    question_cleaned = remove_stopwords_punctuation(question)
    span_cleaned = remove_stopwords_punctuation(span)

    question_bigrams = [(question_cleaned.split()[i], question_cleaned.split()[i+1]) for i in range(len(question_cleaned.split())-1)]
    span_bigrams = [(span_cleaned.split()[i], span_cleaned.split()[i+1]) for i in range(len(span_cleaned.split())-1)]


    question_bigram_counts = Counter(question_bigrams)
    span_bigram_counts = Counter(span_bigrams)

    span_bigrams_set = set(span_bigrams)

    overlap_ratio_sum = 0.0
    for b in question_bigram_counts:
        if b in span_bigrams_set:
            question_bigram_counts[b] = span_bigram_counts[b]
            overlap_ratio = question_bigram_counts[b] / len(span_bigrams)
            overlap_ratio_sum += overlap_ratio

    return overlap_ratio_sum


def get_trigram_overlap(question, span):
    question_cleaned = remove_stopwords_punctuation(question)
    span_cleaned = remove_stopwords_punctuation(span)

    question_trigrams = [(question_cleaned.split()[i], question_cleaned.split()[i+1], question_cleaned.split()[i+2]) for i in range(len(question_cleaned.split())-2)]

    span_trigrams = [(span_cleaned.split()[i], span_cleaned.split()[i+1], span_cleaned.split()[i+2]) for i in range(len(span_cleaned.split())-2)]


    question_trigram_counts = Counter(question_trigrams)
    span_trigram_counts = Counter(span_trigrams)

    span_trigrams_set = set(span_trigrams)

    overlap_ratio_sum = 0.0
    for b in question_trigram_counts:
        if b in span_trigrams_set:
            question_trigram_counts[b] = span_trigram_counts[b]
            overlap_ratio = question_trigram_counts[b] / len(span_trigrams)
            overlap_ratio_sum += overlap_ratio

    return overlap_ratio_sum

def get_bigram_TFIDF(text):
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
    filtered_words = ''.join([c for c in text if c not in string.punctuation])
    filtered_words = ' '.join([word.lower() for word in filtered_words.split() if word.lower() not in stop_words])

    tfidf = TfidfVectorizer(ngram_range=(2,2))
    tfidf_matrix = tfidf.fit_transform([filtered_words])
    feature_names = tfidf.get_feature_names_out()

    if len(feature_names) == 0:
        tfidf_sum = 0
    else:
        tfidf_sum = tfidf_matrix.sum()
    return tfidf_sum


def get_trigram_TFIDF(text):
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
    filtered_words = ''.join([c for c in text if c not in string.punctuation])
    filtered_words = ' '.join([word.lower() for word in filtered_words.split() if word.lower() not in stop_words])

    tfidf = TfidfVectorizer(ngram_range=(3,3))
    tfidf_matrix = tfidf.fit_transform([filtered_words])
    feature_names = tfidf.get_feature_names_out()

    if len(feature_names) == 0:
        tfidf_sum = 0
    else:
        tfidf_sum = tfidf_matrix.sum()
    return tfidf_sum


def get_Minkowski_distance(s1, s2, p=1):
    words1 = s1.split()
    words2 = s2.split()

    freq1 = Counter(words1)
    freq2 = Counter(words2)

    unique_words = set(words1 + words2)

    vec1 = [freq1.get(word, 0) for word in unique_words]
    vec2 = [freq2.get(word, 0) for word in unique_words]

    return distance.minkowski(vec1, vec2, p=p)


def get_Manhattan_distance(s1, s2):
    words1 = s1.split()
    words2 = s2.split()

    freq1 = Counter(words1)
    freq2 = Counter(words2)

    unique_words = set(words1 + words2)

    distance = 0
    for word in unique_words:
        distance += abs(freq1.get(word, 0) - freq2.get(word, 0))

    return distance


def get_Euclidean_distance(s1, s2):
    words1 = s1.split()
    words2 = s2.split()

    freq1 = Counter(words1)
    freq2 = Counter(words2)

    unique_words = set(words1 + words2)

    vec1 = [freq1.get(word, 0) for word in unique_words]
    vec2 = [freq2.get(word, 0) for word in unique_words]

    return distance.euclidean(vec1, vec2)


def get_Hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return -1

    hamming_distance = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            hamming_distance += 1

    return hamming_distance


def get_Jaccard_distance(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_distance = 1 - intersection / union

    return jaccard_distance


def get_constituency_parse(span):
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")
    doc = nlp(span)
    sentences = doc.sentences
    for sentence in sentences:
        return sentence.constituency


def get_POS_tags(span):
    tokens = word_tokenize(span)
    pos = pos_tag(tokens, tagset="universal")
    return pos


def get_length(text):
    return len(text.split())


def get_features(question, span, answer, titleNo, paragNo):
    # # answer_types = get_answer_types(data_answers)      ########
    # syntatic_divergence = get_syntatic_div(question, span, answer)
    # matching_word_frequency = get_matching_word_frequency(question, span)
    bigram_overlap = get_bigram_overlap(question, span)
    trigram_overlap = get_trigram_overlap(question, span)
    # root_match = get_root_matching(question, span)
    # span_length = get_length(span)
    # question_length = get_length(question)
    # span_word_frequency = get_span_TFIDF(span)
    # span_TFIDF = get_span_TFIDF(span)
    bigram_TFIDF = get_bigram_TFIDF(span)
    trigram_TFIDF = get_trigram_TFIDF(span)
    # bm25 = get_BM25()      ########
    # consistant_label = get_constituency_parse(span)
    # span_POS_tags = get_POS_tags(span)
    # hamming_distance = get_Hamming_distance(question, span)
    # jaccard_distance = get_Jaccard_distance(question, span)
    # euclidean_distance = get_Euclidean_distance(question, span)
    # manhattan_distance = get_Manhattan_distance(question, span)
    # minkowski_distance = get_Minkowski_distance(question, span)

    print(bigram_overlap)
    print(trigram_overlap)
    print(bigram_TFIDF)
    print(trigram_TFIDF)
    # features_data = {
    #     "paragNo": titleNo,
    #     "titleNo": paragNo,
    #     "question": question,
    #     "span": span,
    #     "answer": answer,
    #     "syntatic_divergence": syntatic_divergence,
    #     "root_matching": root_match,
    #     "span_TFIDF": span_TFIDF,
    #     "matching_word_frequency": matching_word_frequency,
    #     # "bigram_overlap": bigram_overlap,
    #     # "trigram_overlap": trigram_overlap,
    #     "span_word_frequency": span_word_frequency,
    #     # "bigram_TFIDF": bigram_TFIDF,
    #     # "trigram_TFIDF": trigram_TFIDF,
    #     "minkowski_distance": minkowski_distance,
    #     "manhattan_distance": manhattan_distance,
    #     "euclidean_distance": euclidean_distance,
    #     "hamming_distance": hamming_distance,
    #     "jaccard_distance": jaccard_distance,
    #     "consistant_labels": consistant_label,
    #     "span_POS_tags": span_POS_tags,
    #     "span_length": span_length,
    #     "question_length": question_length,
    # }

    # return features_data


def main():
    data_questions = []
    data_answers = []
    data_spans = []
    data_titleNo = []
    data_paragNo = []
    dict_datas = []

    with open("CSV-Files/devSplit/dev1.json") as f:
        data = json.load(f)

    print(Fore.RED + "Extracting data from dataset...\n")
    for i in range(len(data["data"])):
        for j in range(len(data["data"][i]["paragraphs"])):
            for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                data_spans.append(data["data"][i]["paragraphs"][j]["context"])
                data_questions.append(
                    data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                )
                data_answers.append(
                    data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                )
                data_titleNo.append(i)
                data_paragNo.append(j)

        print(Fore.RED + "For each question and answer extracting features...\n")
        for index in range(len(data_questions)):
            question = data_questions[index]
            span = data_spans[index]
            answer = data_answers[index]
            titleNo = data_titleNo[index]
            paragNo = data_paragNo[index]
            this_result = get_features(question, span, answer, titleNo, paragNo)
            dict_datas.append(this_result)

    # csv_file = "Features/Features_CSV.csv"
    # csv_columns = [
    #     "paragNo",
    #     "titleNo",
    #     "question",
    #     "span",
    #     "answer",
    #     "syntatic_divergence",
    #     "root_matching",
    #     "span_TFIDF",
    #     "matching_word_frequency",
    #     # "bigram_overlap",
    #     # "trigram_overlap",
    #     "span_word_frequency",
    #     # "bigram_TFIDF",
    #     # "trigram_TFIDF",
    #     "minkowski_distance",
    #     "manhattan_distance",
    #     "euclidean_distance",
    #     "hamming_distance",
    #     "jaccard_distance",
    #     "consistant_labels",
    #     "span_POS_tags",
    #     "span_length",
    #     "question_length",
    # ]
    # try:
    #     with open(csv_file, "w") as features_file:
    #         writer = csv.DictWriter(features_file, fieldnames=csv_columns)
    #         writer.writeheader()
    #         for data in dict_datas:
    #             writer.writerow(data)
    # except IOError:
    #     print(Fore.RED + "I/O error")


main()
end_time = time.time()
print(Fore.RED + "Execution Time = ", end_time - start_time)
