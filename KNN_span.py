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
    for i in range(len(data["data"])):
        for j in range(len(data["data"][i]["paragraphs"])):
            for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                if j == len(data["data"][i]["paragraphs"]) - 1:
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

            if j != len(data["data"][i]["paragraphs"]) - 1:
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


def train_find_answer_sentence(data_train_list):
    for test in data_train_list:
        paragraph = test[0][0]
        train_data_questions = test[1]
        train_data_answers = test[2]
        train_data_titleNo = test[3]
        train_data_paragNo = test[4]

        sentences = nltk.sent_tokenize(paragraph)
        answer_spans = []
        for answer in train_data_answers:
            condidate_spans = []
            for sentence in sentences:
                if answer in sentence:
                    condidate_spans.append(sentence)

            answer_spans.append(
                [
                    train_data_questions[train_data_answers.index(answer)],
                    answer,
                    condidate_spans,
                ]
            )

    return answer_spans


def test_features(data_test_list):
    pass


def train_features(data_train_list):
    pass


def test_condidate_sentences(answer_spans, data_test_list):
    test_features(data_test_list)


def main():
    data_list = split_train_test()
    data_train_list = data_list[0]
    data_test_list = data_list[1]
    answer_spans = train_find_answer_sentence(data_train_list)
    # test_features(data_test_list)
    train_features(data_train_list)
    test_condidate_sentences(answer_spans, data_test_list)

    # for item in condidate_answers:
    #     print(item)


main()
end_time = time.time()
print(Fore.RED + "Execution Time = ", (end_time - start_time) / 60)
