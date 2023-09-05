import spacy
import pandas as pd
from nltk.stem import WordNetLemmatizer
import datetime
from colorama import Fore
import time
import numpy as np
from sklearn.model_selection import train_test_split
import json
import nltk
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

start_time = time.time()
print(
    Fore.WHITE
    + "Start Time: = %s:%s:%s\n"
    % (
        datetime.datetime.now().hour,
        datetime.datetime.now().minute,
        datetime.datetime.now().second,
    )
)

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# print(Fore.RED + "Dataset : CSV-Files/devSplit/dev1.json\n")
# with open("CSV-Files/devSplit/dev1.json") as f:
#     data = json.load(f)

##############################################################################################################
##############################################################################################################

train_data = []
test_data = []


def split_train_test():
    with open("CSV-Files/devSplit/dev2.json") as f:
        data = json.load(f)

    print(Fore.RED + "Extracting data from dataset...\n")

    for i in range(len(data["data"])):
        train_data_questions = []
        train_data_parags = []
        train_data_answers = []
        train_data_spans = []
        train_data_titleNo = []
        train_data_paragNo = []

        # test_data_questions = []
        # test_data_answers = []
        # test_data_spans = []
        # test_data_titleNo = []
        # test_data_paragNo = []
        # for each paragraph
        for j in range(len(data["data"][i]["paragraphs"])):
            # for each context
            for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                # if j == len(data["data"][i]["paragraphs"]) - 2:
                # test_data_spans.append(data["data"][i]["paragraphs"][j]["context"])
                # test_data_questions.append(
                #     data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                # )
                # test_data_answers.append(
                #     data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                # )
                # test_data_titleNo.append(i)
                # test_data_paragNo.append(j)

                # else:
                train_data_parags.append(data["data"][i]["paragraphs"][j]["context"])
                train_data_questions.append(
                    data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                )
                train_data_answers.append(
                    data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                )
                train_data_titleNo.append(i)
                train_data_paragNo.append(j)

                train_spans = nltk.sent_tokenize(
                    data["data"][i]["paragraphs"][j]["context"]
                )
                flag = 0
                answer = data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0][
                    "text"
                ]
                for sentence in train_spans:
                    if sentence.lower().count(answer.lower()) != 0:
                        train_data_spans.append(sentence)
                        flag = 1
                        break

                if flag == 0:
                    train_data_spans.append("NaN")

    train_data.append(
        [
            train_data_parags,
            train_data_spans,
            train_data_questions,
            train_data_answers,
            train_data_titleNo,
            train_data_paragNo,
        ]
    )
    # test_data.append(
    #     [
    #         test_data_spans,
    #         test_data_questions,
    #         test_data_answers,
    #         test_data_titleNo,
    #         test_data_paragNo,
    #     ]
    # )

    train_df = {
        "context": train_data[0][0],
        "span": train_data[0][1],
        "question": train_data[0][2],
        "answer": train_data[0][3],
        "titleNo": train_data[0][4],
        "paragraphNo": train_data[0][5],
    }

    # test_spans = nltk.sent_tokenize(test_data[0][0][0])
    # t_span = []
    # t_question = []
    # t_answer = []
    # t_title = []
    # t_parag = []
    # for sentence in test_spans:
    #     for index in range(len(test_data[0][1])):
    #         t_span.append(sentence)
    #         t_question.append(test_data[0][1][index])
    #         t_answer.append(test_data[0][2][index])
    #         t_title.append(test_data[0][3][index])
    #         t_parag.append(test_data[0][4][index])

    # test_df = {
    #     "span": t_span,
    #     "question": t_question,
    #     "answer": t_answer,
    #     "titleNo": t_title,
    #     "paragraphNo": t_parag,
    # }
    train_df = pd.DataFrame(train_df)
    # test_df = pd.DataFrame(test_df)
    train_df.to_csv("ClassificationAlgos/DF_dev2.csv", index=False)
    # test_df.to_csv("ClassificationAlgos/TestData_dev1_p52.csv", index=False)


def get_features():
    df = pd.read_csv("ClassificationAlgos/Features_dev2.csv", encoding="utf8")

    # Remove rows with missing values
    df = df.dropna()
    ########################################################
    # Declare feature vector and target variable
    X = df.drop(["span"], axis="columns")
    y = df["span"]

    # Split data into separate training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=0
    )
    X_train_copy, X_test_copy, y_train_copy, y_test_copy = (
        X_train,
        X_test,
        y_train,
        y_test,
    )

    #######################################################
    categorical = [col for col in X_train.columns if X_train[col].dtypes == "O"]
    print(Fore.GREEN + "Categorical Variables:", categorical)
    numerical = [col for col in X_train.columns if X_train[col].dtypes != "O"]
    print(Fore.GREEN + "Numerical Variables:", numerical)

    ########################################################
    # impute missing categorical variables with most frequent value

    for df2 in [X_train, X_test]:
        df2["wh_word"].fillna(X_train["wh_word"].mode()[0], inplace=True)
        df2["syntactic_divergence"].fillna(
            X_train["syntactic_divergence"].mode()[0], inplace=True
        )
        df2["root_matching"].fillna(X_train["root_matching"].mode()[0], inplace=True)
        df2["span_TFIDF"].fillna(X_train["span_TFIDF"].mode()[0], inplace=True)
        df2["matching_word_frequency"].fillna(
            X_train["matching_word_frequency"].mode()[0], inplace=True
        )
        df2["bigram_overlap"].fillna(X_train["bigram_overlap"].mode()[0], inplace=True)
        df2["trigram_overlap"].fillna(
            X_train["trigram_overlap"].mode()[0], inplace=True
        )
        df2["span_word_frequency"].fillna(
            X_train["span_word_frequency"].mode()[0], inplace=True
        )
        df2["bigram_TFIDF"].fillna(X_train["bigram_TFIDF"].mode()[0], inplace=True)
        df2["trigram_TFIDF"].fillna(X_train["trigram_TFIDF"].mode()[0], inplace=True)
        df2["minkowski_distance"].fillna(
            X_train["minkowski_distance"].mode()[0], inplace=True
        )
        df2["manhattan_distance"].fillna(
            X_train["manhattan_distance"].mode()[0], inplace=True
        )
        df2["euclidean_distance"].fillna(
            X_train["euclidean_distance"].mode()[0], inplace=True
        )
        df2["hamming_distance"].fillna(
            X_train["hamming_distance"].mode()[0], inplace=True
        )
        df2["jaccard_distance"].fillna(
            X_train["jaccard_distance"].mode()[0], inplace=True
        )
        df2["edit_distance"].fillna(X_train["edit_distance"].mode()[0], inplace=True)
        df2["span_length"].fillna(X_train["span_length"].mode()[0], inplace=True)
        # df2["question_length"].fillna(X_train["question_length"].mode()[0], inplace=True)

    ######################################################
    # encode remaining variables with one-hot encoding
    X_train_question = X_train["question"]
    X_train_answer = X_train["answer"]
    X_test_question = X_test["question"]
    X_test_answer = X_test["answer"]
    Y_train_span = y_train_copy.tolist()
    Y_test_span = y_test_copy.tolist()
    encoder = ce.OneHotEncoder(
        cols=[
            "question",
            # "span",
            "answer",
        ]
    )
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    ######################################################
    # Feature Scaling
    cols = X_train.columns
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    ######################################################
    # train a Gaussian Naive Bayes classifier on the training set
    # instantiate the model
    gnb = GaussianNB()
    # fit the model
    gnb.fit(X_train, y_train)

    ######################################################
    # Predict the Results
    y_pred = gnb.predict(X_test)

    # y_list = []
    # y_list.extend(y_train.tolist())
    # y_list.extend(y_pred.tolist())

    # accuracy_list = []
    # for item_index in range(len(y_list)):
    #     if y_list[item_index].lower() == df["span"].tolist()[item_index].lower():
    #         accuracy_list.append(True)
    #     else:
    #         accuracy_list.append(False)

    # print(len(Fore.RED + str(y_list)))

    ######################################################
    # Model Accuracy
    print(
        Fore.LIGHTBLUE_EX
        + "\nModel accuracy score: {0:0.4f}%".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

    ######################################################
    train_questions = pd.get_dummies(X_train_question).idxmax(1).tolist()
    train_answers = pd.get_dummies(X_train_answer).idxmax(1).tolist()
    test_questions = pd.get_dummies(X_test_question).idxmax(1).tolist()
    test_answers = pd.get_dummies(X_test_answer).idxmax(1).tolist()

    train_newData = {
        "titleNo": X_train_copy["titleNo"],
        "paragNo": X_train_copy["paragNo"],
        "question": train_questions,
        "span": Y_train_span,
        "answer": train_answers,
        "wh_word": X_train_copy["wh_word"],
        "syntactic_divergence": X_train_copy["syntactic_divergence"],
        "root_matching": X_train_copy["root_matching"],
        "span_TFIDF": X_train_copy["span_TFIDF"],
        "matching_word_frequency": X_train_copy["matching_word_frequency"],
        "bigram_overlap": X_train_copy["bigram_overlap"],
        "trigram_overlap": X_train_copy["trigram_overlap"],
        "span_word_frequency": X_train_copy["span_word_frequency"],
        "bigram_TFIDF": X_train_copy["bigram_TFIDF"],
        "trigram_TFIDF": X_train_copy["trigram_TFIDF"],
        "minkowski_distance": X_train_copy["minkowski_distance"],
        "manhattan_distance": X_train_copy["manhattan_distance"],
        "euclidean_distance": X_train_copy["euclidean_distance"],
        "hamming_distance": X_train_copy["hamming_distance"],
        "jaccard_distance": X_train_copy["jaccard_distance"],
        "edit_distance": X_train_copy["edit_distance"],
        "span_length": X_train_copy["span_length"],
        "question_length": X_train_copy["question_length"],
    }
    newDF = pd.DataFrame(train_newData)
    newDF.to_csv("ClassificationAlgos/Train_NB_dev2.csv", index=False)

    is_accurate_list = [[] for _ in range(len(test_questions))]
    real_span_list = [[] for _ in range(len(test_questions))]
    test_newData = {
        "titleNo": X_test_copy["titleNo"],
        "paragNo": X_test_copy["paragNo"],
        "question": test_questions,
        "predict_span": y_pred,
        "real_span": real_span_list,
        "is_accurate": is_accurate_list,
        "answer": test_answers,
        "wh_word": X_test_copy["wh_word"],
        "syntactic_divergence": X_test_copy["syntactic_divergence"],
        "root_matching": X_test_copy["root_matching"],
        "span_TFIDF": X_test_copy["span_TFIDF"],
        "matching_word_frequency": X_test_copy["matching_word_frequency"],
        "bigram_overlap": X_test_copy["bigram_overlap"],
        "trigram_overlap": X_test_copy["trigram_overlap"],
        "span_word_frequency": X_test_copy["span_word_frequency"],
        "bigram_TFIDF": X_test_copy["bigram_TFIDF"],
        "trigram_TFIDF": X_test_copy["trigram_TFIDF"],
        "minkowski_distance": X_test_copy["minkowski_distance"],
        "manhattan_distance": X_test_copy["manhattan_distance"],
        "euclidean_distance": X_test_copy["euclidean_distance"],
        "hamming_distance": X_test_copy["hamming_distance"],
        "jaccard_distance": X_test_copy["jaccard_distance"],
        "edit_distance": X_test_copy["edit_distance"],
        "span_length": X_test_copy["span_length"],
        "question_length": X_test_copy["question_length"],
    }
    newDF = pd.DataFrame(test_newData)
    newDF.to_csv("ClassificationAlgos/Test_NB_dev2.csv", index=False)
    #####################################################
    # print the scores on training and test set
    print("\nTraining set score: {:.4f}%".format(gnb.score(X_train, y_train) * 100))
    print("Test set score: {:.4f}%".format(gnb.score(X_test, y_test) * 100))
    with open("ClassificationAlgos/results.txt", "a") as text_file:
        text_file.write(
            "\nNB : Model accuracy score: {0:0.4f}%".format(
                accuracy_score(y_test, y_pred) * 100
            )
            + "\n"
        )

    return


def change_accuracy_realSpan():
    df = pd.read_csv("ClassificationAlgos/Features_dev2.csv", encoding="utf8")
    test_df = pd.read_csv("ClassificationAlgos/Test_NB_dev2.csv", encoding="utf8")

    test_predict_spans = test_df["predict_span"].tolist()
    test_questions = test_df["question"].tolist()
    test_answers = test_df["answer"].tolist()
    test_paragNos = test_df["paragNo"].tolist()
    original_spans = df["span"].tolist()
    original_questions = df["question"].tolist()
    original_answers = df["answer"].tolist()
    original_paragNos = df["paragNo"].tolist()

    real_spans = []
    is_accurate = []
    for index in range(len(test_questions)):
        question = test_questions[index]
        answer = test_answers[index]
        paragNo = test_paragNos[index]

        for original_index in range(len(original_spans)):
            origin_question = original_questions[original_index]
            origin_answer = original_answers[original_index]
            origin_paragNo = original_paragNos[original_index]

            if (
                (origin_question == question)
                and (origin_answer == answer)
                and (origin_paragNo == paragNo)
            ):
                span = original_spans[original_index]
                real_spans.append(span)

                if span == test_predict_spans[index]:
                    is_accurate.append(True)

                else:
                    is_accurate.append(False)

                break

    test_df["real_span"] = real_spans
    test_df["is_accurate"] = is_accurate
    test_df.to_csv("ClassificationAlgos/newTest_NB_dev2_.csv", index=False)
    return


def main():
    # split_train_test()
    get_features()
    change_accuracy_realSpan()


main()
end_time = time.time()
print(Fore.WHITE + "\nExecution Time = ", (end_time - start_time), "seconds")
