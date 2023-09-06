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
import string
from csv import writer
import datetime
from colorama import Fore
from nltk.tokenize import TreebankWordTokenizer
import time
import math
import random


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


nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

print(Fore.RED + "Dataset : CSV-Files/devSplit/dev1.json\n")
with open("CSV-Files/devSplit/dev1.json") as f:
    data = json.load(f)

##############################################################################################################
##############################################################################################################

train_data = []
test_data = []


def split_train_test():
    with open("CSV-Files/devSplit/dev1.json") as f:
        data = json.load(f)

    print(Fore.RED + "Extracting data from dataset...\n")

    for i in range(len(data["data"])):
        train_data_questions = []
        train_data_answers = []
        train_data_spans = []
        train_data_titleNo = []
        train_data_paragNo = []

        test_data_questions = []
        test_data_answers = []
        test_data_spans = []
        test_data_titleNo = []
        test_data_paragNo = []
        # for each paragraph
        for j in range(len(data["data"][i]["paragraphs"])):
            # for each context
            for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                if j == len(data["data"][i]["paragraphs"]) - 2:
                    test_data_spans.append(data["data"][i]["paragraphs"][j]["context"])
                    test_data_questions.append(
                        data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                    )
                    test_data_answers.append(
                        data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                    )
                    test_data_titleNo.append(i)
                    test_data_paragNo.append(j)

                else:
                    train_data_spans.append(data["data"][i]["paragraphs"][j]["context"])
                    train_data_questions.append(
                        data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                    )
                    train_data_answers.append(
                        data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                    )
                    train_data_titleNo.append(i)
                    train_data_paragNo.append(j)

    train_data.append(
        [
            train_data_spans,
            train_data_questions,
            train_data_answers,
            train_data_titleNo,
            train_data_paragNo,
        ]
    )
    test_data.append(
        [
            test_data_spans,
            test_data_questions,
            test_data_answers,
            test_data_titleNo,
            test_data_paragNo,
        ]
    )

    return (train_data, test_data)

    # print("Creating Test_Data and Train_Data files...")
    # trainData = {
    #     "question": train_data_questions,
    #     "answer": train_data_answers,
    #     "span": train_data_spans,
    #     "paragraphNo": train_data_paragNo,
    #     "titleNo": train_data_titleNo,
    # }

    # df = pd.DataFrame(trainData)
    # df.to_csv("KNN/TrainData_dev1.csv", encoding="utf-8", index=False)

    # testData = {
    #     "question": test_data_questions,
    #     "answer": test_data_answers,
    #     "span": test_data_spans,
    #     "paragraphNo": test_data_paragNo,
    #     "titleNo": test_data_titleNo,
    # }

    # df = pd.DataFrame(testData)
    # df.to_csv("KNN/TestData_dev1.csv", encoding="utf-8", index=False)


# def find_answer_sentence_test(data_test_list):
#     for test in data_test_list:
#         paragraph = test[0][0]
#         test_data_questions = test[1]
#         test_data_answers = test[2]
#         test_data_titleNo = test[3]
#         test_data_paragNo = test[4]

#         sentences = nltk.sent_tokenize(paragraph)
#         answer_spans = []
#         for answer in test_data_answers:
#             condidate_spans = []
#             for sentence in sentences:
#                 if answer in sentence:
#                     condidate_spans.append(sentence)

#             answer_spans.append(
#                 [
#                     test_data_questions[test_data_answers.index(answer)],
#                     answer,
#                     condidate_spans,
#                 ]
#             )

#     return answer_spans


# def remove_stopword(text):
#     stop_words = set(
#         stopwords.words("english")
#         + [
#             "though",
#             "and",
#             "I",
#             "A",
#             "a",
#             "an",
#             "An",
#             "And",
#             "So",
#             ".",
#             ",",
#             ")",
#             "By",
#             "(",
#             "''",
#             "Other",
#             "The",
#             ";",
#             "however",
#             "still",
#             "the",
#             "They",
#             "For",
#             "for",
#             "also",
#             "In",
#             "This",
#             "When",
#             "It",
#             "so",
#             "Yes",
#             "yes",
#             "No",
#             "no",
#             "These",
#             "these",
#             "This",
#         ]
#     )
#     filtered_words = "".join([c for c in text if c not in string.punctuation])
#     filtered_words = " ".join(
#         [
#             word.lower()
#             for word in filtered_words.split()
#             if word.lower() not in stop_words
#         ]
#     )
#     return filtered_words


# def condidate_answers_test(answer_spans_list):
#     all_condidate_answers = []
#     question_list = []
#     sentence_list = []
#     for item in answer_spans_list:
#         question = item[0]
#         answer = item[1]
#         condidate_spans = item[2]

#         condidate_answers = []
#         for sentence in condidate_spans:
#             if len(condidate_answers) == 0:
#                 condidate_answers.append(answer)

#             filtered_words = remove_stopword(sentence).split()
#             while len(condidate_answers) <= 5:
#                 random_answer = random.choice(filtered_words)

#                 if condidate_answers.count(random_answer) == 0:
#                     for ans in condidate_answers:
#                         if random_answer.lower().count(ans) == 0:
#                             condidate_answers.append(random_answer)
#                             break

#         question_list.append(question)
#         sentence_list.append(sentence)
#         all_condidate_answers.append(condidate_answers)

#     condidate_ans_data = {
#         "question": question_list,
#         "span": sentence_list,
#         "condidate_answers": all_condidate_answers,
#     }

#     df = pd.DataFrame(condidate_ans_data)
#     df.to_csv("KNN/CondidateAnswers_dev1.csv", encoding="utf-8", index=False)

# return all_condidate_answers


def main():
    data_list = split_train_test()
    data_train_list = data_list[0]
    data_test_list = data_list[1]
    result_train_spans = []
    result_train_questions = []
    result_train_answers = []
    result_train_titleNos = []
    result_train_paragraphNos = []

    for span_index in range(len(data_train_list[0][0])):
        train_spans = nltk.sent_tokenize(data_train_list[0][0][span_index])
        for sentence in train_spans:
            if data_train_list[0][2][span_index] in sentence:
                result_train_spans.append(sentence)
                result_train_questions.append(data_train_list[0][1][span_index])
                result_train_answers.append(data_train_list[0][2][span_index])
                result_train_titleNos.append(data_train_list[0][3][span_index])
                result_train_paragraphNos.append(data_train_list[0][4][span_index])
    train_df = {
        "span": result_train_spans,
        "question": result_train_answers,
        "answer": result_train_answers,
        "titleNo": result_train_titleNos,
        "paragraphNo": result_train_paragraphNos,
    }

    test_spans = nltk.sent_tokenize(data_test_list[0][0][0])
    # print(test_spans)
    t_span = []
    t_question = []
    t_answer = []
    t_title = []
    t_parag = []
    for sentence in test_spans:
        for index in range(len(data_test_list[0][1])):
            answer = data_test_list[0][2][index]
            if answer in sentence:
                t_span.append(sentence)
                t_question.append(data_test_list[0][1][index])
                t_answer.append(data_test_list[0][2][index])
                t_title.append(data_test_list[0][3][index])
                t_parag.append(data_test_list[0][4][index])

    test_df = {
        "span": t_span,
        "question": t_question,
        "answer": t_answer,
        "titleNo": t_title,
        "paragraphNo": t_parag,
    }
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    train_df.to_csv("KNN/TrainData_dev1_p522.csv", index=False)
    test_df.to_csv("KNN/TestData_dev1_p522.csv", index=False)

    # train_question = data_train_list[1][1]

    # print(len(data_train_list[1][1]))
    # answer_spans = find_answer_sentence_test(data_test_list)
    # condidate_answers_test(answer_spans)

    # for item in condidate_answers:
    #     print(item)


main()
end_time = time.time()
print(Fore.RED + "Execution Time = ", (end_time - start_time) / 60)
