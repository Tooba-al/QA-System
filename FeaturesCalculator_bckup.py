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

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

with open("CSV-Files/devSplit/dev1.json") as f:
    data = json.load(f)


# data_questions = data['question']
# data_answers = data['answer']
# data_parags = data['paragNo']
# data_titles = data['titleNo']
# for i in range(len(data['data'])):
#     for j in range(len(data['data'][i]['paragraphs'])):
#         # answers = []
#         # questions = []
#         for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
#             data_questions.append(
#                 data['data'][i]['paragraphs'][j]['qas'][k]['question'])
#             data_answers.append(
#                 data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])

#         # data_questions.append(questions)
#         # data_answers.append(answers)
#             data_parags.append(j)
#             data_titles.append(0)

# newData = {
#     "question": data_questions,
#     "answer": data_answers,
#     "paragNo": data_parags,
#     "titleNo": data_titles,
# }

# df = pd.DataFrame(newData)
# df.to_csv('Features/question_answer_dev1.csv',
#           encoding='utf-8', index=False)

##############################################################################################################
##############################################################################################################

# data = pd.read_csv("Features/question_sentence.csv")
# question_list = data["question"].tolist()
# sentence_list = data["sentence"].tolist()


# def word_lemmma(sentence, question):
#     sent_lemma = []
#     ques_lemma = []

#     for word in sentence.split():
#         sent_lemma.append(lemmatizer.lemmatize(word))

#     for word in question.split():
#         ques_lemma.append(lemmatizer.lemmatize(word))

#     edit_distance = 0
#     for word in sent_lemma:
#         if word in ques_lemma:
#             edit_distance += 1

#     return edit_distance


# wordLemma_list = []
# for index in range(len(question_list)):
#     ED = word_lemmma(question_list[index], sentence_list[index])
#     wordLemma_list.append(ED)

# newData = {
#     "question": question_list,
#     "sentence": sentence_list,
#     "word_lemma": wordLemma_list,
#     "paragraphNo": data["paragraphNo"].tolist(),
#     "titleNo": data["titleNo"].tolist(),
# }

# df = pd.DataFrame(newData)
# df.to_csv('Features/WordLemma_dev1.csv', encoding='utf-8', index=False)

##############################################################################################################
##############################################################################################################


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


# def get_answer_types(answersList):
#     ans = []
#     ques = []
#     # cardinals = []
#     # person = []
#     # date = []
#     # loc = []
#     # money = []
#     # gpe = []
#     # org = []
#     # event = []
#     # other = []
#     labels = []
#     types = []

#     NER = spacy.load("en_core_web_sm")
#     for answer_title in answersList:
#         for answer in answer_title:
#             text2 = NER(answer)
#             for word in text2.ents:
#                 if (word.label_ == "CARDINAL"):
#                     types.append("numerical")
#                     # cardinals.append(word.text)
#                 # elif (word.label_ == "PERSON"):
#                 #     types.append("non-numerical")
#                 #     person.append(word.text)
#                 # elif (word.label_ == "DATE"):
#                 #     types.append("non-numerical")
#                 #     date.append(word.text)
#                 # elif (word.label_ == "LOC"):
#                 #     types.append("non-numerical")
#                 #     loc.append(word.text)
#                 # elif (word.label_ == "MONEY"):
#                 #     types.append("non-numerical")
#                 #     money.append(word.text)
#                 # elif (word.label_ == "ORG"):
#                 #     types.append("non-numerical")
#                 #     org.append(word.text)
#                 # elif (word.label_ == "GPE"):
#                 #     types.append("non-numerical")
#                 #     gpe.append(word.text)
#                 # elif (word.label_ == "EVENT"):
#                 #     types.append("non-numerical")
#                 #     event.append(word.text)
#                 else:
#                     types.append("non-numerical")
#                     # other.append(word.text)

#                 labels.append(word.text)
#                 ans.append(answer)
#                 print(len(answersList))
#                 print(len(answersList[0]))
#                 index = answersList[0].index(
#                     answer_title.index(answer))
#                 print(index)
#                 ques.append(data_questions[index])

#     print(len(ques))
#     print(len(ans))
#     print(len(labels))
#     print(len(types))
#     newData = {
#         "question": ques,
#         "answer": ans,
#         "label": labels,
#         "type": types,
#     }

#     df = pd.DataFrame(newData)
#     df.to_csv('Features/AnswerTypes_dev1.csv',
#               encoding='utf-8', index=False)


def get_reasoning_types(ans):
    pass


def dependency_parser(span):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(span)
    # position_label = []
    token_positions = []
    # children = []

    # to see all the children token
    for token in doc:
        #     token_childs = []
        #     position_label.append(
        #         (
        #             str(token.dep_),
        #             str(token.head.text) + "-" + str(token.head.i),
        #             str(token.text) + "-" + str(token.i),
        #         )
        #     )
        #     childs = [str(t.text) + "-" + str(t.i) for t in token.children]
        #     token_childs = [str(token) + "-" + str(token.i), childs]
        #     children.append(token_childs)
        token_positions.append(str(token) + "-" + str(token.i))

    # root = ""
    # for token in doc:
    #     if token.dep_ == "ROOT":
    #         root = str(token) + "-" + str(token.i)
    #         break

    return token_positions
    # return (root, position_label, children)


def shortest_dependency_path(src, dest, text):
    tokens_position = dependency_parser(text)

    document = nlp(text)

    edges = []
    for token in document:
        for child in token.children:
            edges.append(
                (
                    "{0}-{1}".format(token.lower_, token.i),
                    "{0}-{1}".format(child.lower_, child.i),
                )
            )

    src_token = ""
    dest_token = ""
    for token in tokens_position:
        if src in token:
            src_token = token
        elif dest in token:
            dest_token = token

    graph = nx.Graph(edges)
    path_length = nx.shortest_path_length(graph, source=src_token, target=dest_token)
    path = nx.shortest_path(graph, source=src_token, target=dest_token)

    return (path_length, path)


def anchors(question_list, sentence_list, parag_list, title_list):
    question_index = []
    sentence_index = []
    common_word = []
    paragraph_no = []
    title_no = []
    questions_list = []
    sentences_list = []

    def extractAnchors(question, sentence, paragraphNo, titleNo):
        question_words = stopword_func(question)
        sentence_words = stopword_func(sentence)

        for q_word in question_words:
            for s_word in sentence_words:
                qq_word = lemmatizer.lemmatize(q_word)
                ss_word = lemmatizer.lemmatize(s_word)

                if qq_word == ss_word:
                    questions_list.append(question)
                    sentences_list.append(sentence)
                    question_index.append(question_words.index(q_word))
                    sentence_index.append(sentence_words.index(s_word))
                    common_word.append(q_word)
                    paragraph_no.append(paragraphNo)
                    title_no.append(titleNo)

    for index in range(len(question_list)):
        extractAnchors(
            question_list[index],
            sentence_list[index],
            parag_list[index],
            title_list[index],
        )

    newData = {
        "word": common_word,
        "question": questions_list,
        "sentence": sentences_list,
        "question_index": question_index,
        "sentence_index": sentence_index,
        "paragraphNo": paragraph_no,
        "titleNo": title_no,
    }

    df = pd.DataFrame(newData)
    df.to_csv("Features/Anchors_dev1.csv", encoding="utf-8", index=False)


def find_answer_sentence(paragraph_no, answer):
    contexts = data["data"][0]["paragraphs"][paragraph_no]["context"]
    sentences = nltk.sent_tokenize(contexts)
    result = []
    for sentence in sentences:
        if answer in sentence:
            result.append(sentence)

    return result


def get_syntatic_div(question):
    data_qa = pd.read_csv("Features/question_answer_dev1.csv")
    data_qs = pd.read_csv("Features/question_sentence_dev1.csv")
    question_list = data_qs["question"].tolist()
    answer_list = data_qa["answer"].tolist()
    sentence_list = data_qs["sentence"].tolist()
    parag_list = data_qs["paragraphNo"].tolist()
    title_list = data_qs["titleNo"].tolist()

    answer = ""
    paragraph_no = 0
    anchors(question_list, sentence_list, parag_list, title_list)
    for q in question_list:
        if q == question:
            answer = answer_list[question_list.index(q)]

    sentence = find_answer_sentence(paragraph_no, answer)

    data_anchor = pd.read_csv("Features/Anchors_dev1.csv")
    anchors = data_anchor["anchor"]
    questions = data_anchor["question"]
    sentences = data_anchor["sentence"]

    for index in range(len(questions)):
        if questions[index] == question and sentences[index] == sentence:
            anchor = anchors[index]
            answer_SDP = shortest_dependency_path(anchor, answer, sentence)
            question_SDP = shortest_dependency_path(question[0], anchor, question)

    return (answer_SDP, question_SDP)
    # for span in sentence_list:
    #     DP = dependency_parser(span)
    #     root = DP[0]
    #     position_label = DP[1]
    #     children = DP[2]


def get_root_matching(question, span):
    # question = ' '.join(question)
    # span = ' '.join(span)

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
    pass


def get_trigram_overlap(question, span):
    pass


def get_bigram_TFIDF(text):
    pass


def get_trigram_TFIDF(text):
    pass


def get_Minkowski_distance():
    pass


def get_Manhattan_distance():
    pass


def get_Euclidean_distance():
    pass


def get_unlexicalized_path(question, span):
    pass


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


def get_length(span):
    return len(span)


def get_features(question, span):
    # question = stopword_func(question)
    # span = stopword_func(span)

    # answer_types = get_answer_types(data_answers)      ########
    # syntatic_divergence = get_syntatic_div(question)  ########
    # lexicalized_feature =       ########
    # matching_word_frequency = get_matching_word_frequency(question, span)
    # biagram_overlap = get_bigram_overlap(span)      ########
    # triagram_overlap = get_trigram_overlap(span)      ########
    # root_match = get_root_matching(question, span)
    # length = get_length(span)
    # span_word_frequency = get_span_TFIDF(span)
    # span_TFIDF = get_span_TFIDF(span)
    # bigram_TFIDF = get_bigram_TFIDF(span)      ########
    # trigram_TFIDF = get_trigram_TFIDF(span)      ########
    # bm25 = get_BM25()      ########
    # consistant_label = get_constituency_parse(span)
    # span_POS_tags = get_POS_tags(span)
    # dependency_tree_path =      ########,0.

    # print(syntatic_divergence)
    pass


span = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season."
question = "Which NFL team represented the AFC at Super Bowl 50?"
get_features(question, span)
