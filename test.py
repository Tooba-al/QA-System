import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import nltk
from nltk import pos_tag
from nltk import word_tokenize
import networkx as nx
from scipy.spatial import distance
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

with open("CSV-Files/devSplit/dev1.json") as f:
    data = json.load(f)


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
    path_length = nx.shortest_path_length(graph, source=wh_token, target=anchor_token)
    path = nx.shortest_path(graph, source=wh_token, target=anchor_token)

    return (path_length, path)


def A_shortest_dependency_path(anchor, answer, text):
    tokens_position = dependency_parser(text)

    document = nlp(text)

    anchor = anchor.lower()
    answer = answer.lower()

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
    path_length = nx.shortest_path_length(
        graph, source=anchor_token, target=answer_token
    )
    path = nx.shortest_path(graph, source=anchor_token, target=answer_token)

    return (path_length, path)


def edit_distance(Q_SDP, A_SDP):
    ed = 0
    if Q_SDP[0] > A_SDP[0]:
        ed = Q_SDP[0] - A_SDP[0]
    else:
        ed = A_SDP[0] - Q_SDP[0]

    return ed


def get_syntatic_div(question, paragNo):
    data_qa = pd.read_csv("Features/question_answer_dev1.csv")
    data_qs = pd.read_csv("Features/question_sentence_dev1.csv")
    question_list = data_qs["question"].tolist()
    answer_list = data_qa["answer"].tolist()
    # sentence_list = data_qs["sentence"].tolist()
    parag_list = data_qs["paragraphNo"].tolist()
    title_list = data_qs["titleNo"].tolist()
    answer = ""
    paragraph_no = 0

    # it took so long
    # calculate_anchors(question_list, sentence_list, parag_list, title_list)
    for q in question_list:
        if q == question:
            answer = answer_list[question_list.index(q)]
            break

    sentence = find_answer_sentence(paragraph_no, answer)
    data_anchor = pd.read_csv("Features/Anchors_dev1.csv")
    tempAnchor = data_anchor.loc[data_anchor["sentence"] == sentence].copy()
    tempAnchors = tempAnchor.loc[tempAnchor["question"] == question].copy()
    anchors = tempAnchors["anchor"].to_list()
    questions = tempAnchors["question"].to_list()
    sentences = tempAnchors["sentence"].to_list()

    answer_SDP = []
    question_SDP = []
    sentence_found = ""
    question_found = ""
    index_found = ""
    for index in range(len(questions)):
        if questions[index] == question:
            if sentences[index] == sentence:
                question_found = question
                sentence_found = sentence
                index_found = index
                break

    anchor_list = []

    [anchor_list.append(item) for item in anchors if item not in anchor_list]
    for anchor in anchor_list:
        question_SDP.append(
            Q_shortest_dependency_path(anchor, question_found.split()[0], question)
        )
        answer_SDP.append(A_shortest_dependency_path(anchor, answer, sentence))

    # print(question_SDP)
    # print(answer_SDP)

    ED_list = []
    for index in range(len(question_SDP)):
        q_SDP = question_SDP[index]
        a_SDP = answer_SDP[index]

        ED = edit_distance(a_SDP, q_SDP)
        ED_list.append(ED)

    min_ED = min(ED_list)

    return (
        min_ED,
        (answer_SDP[ED_list.index(min_ED)], question_SDP[ED_list.index(min_ED)]),
    )


def get_features(question, span):
    # question = stopword_func(question)
    # span = stopword_func(span)

    # answer_types = get_answer_types(data_answers)      ########
    syntatic_divergence = get_syntatic_div(question, 0)
    # lexicalized_feature =       ########
    # matching_word_frequency = get_matching_word_frequency(question, span)
    # biagram_overlap = get_bigram_overlap(span)
    # triagram_overlap = get_trigram_overlap(span)
    # root_match = get_root_matching(question, span)
    # length = get_length(span)
    # span_word_frequency = get_span_TFIDF(span)
    # span_TFIDF = get_span_TFIDF(span)
    # bigram_TFIDF = get_bigram_TFIDF(span)
    # trigram_TFIDF = get_trigram_TFIDF(span)
    # bm25 = get_BM25()      ########
    # consistant_label = get_constituency_parse(span)
    # span_POS_tags = get_POS_tags(span)
    # dependency_tree_path =      ########,0.

    print(syntatic_divergence)


span = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season."
question = "Which NFL team represented the AFC at Super Bowl 50?"
get_features(question, span)
