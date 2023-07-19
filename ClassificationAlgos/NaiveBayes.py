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
    with open("CSV-Files/devSplit/dev1.json") as f:
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
    train_df.to_csv("ClassificationAlgos/DF_dev1.csv", index=False)
    # test_df.to_csv("ClassificationAlgos/TestData_dev1_p52.csv", index=False)


def get_features():
    df = pd.read_csv("ClassificationAlgos/Features_dev1.csv", encoding="utf8")

    ########################################################
    # Declare feature vector and target variable
    X = df.drop(["answer"], axis="columns")
    y = df["answer"]

    # Split data into separate training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=0
    )

    #######################################################
    categorical = [col for col in X_train.columns if X_train[col].dtypes == "O"]
    print(Fore.GREEN + "Categorical Variables:", categorical)
    numerical = [col for col in X_train.columns if X_train[col].dtypes != "O"]
    print(Fore.GREEN + "Numerical Variables:", numerical)

    ########################################################
    # impute missing categorical variables with most frequent value
    for df2 in [X_train, X_test]:
        # df2["question"].fillna(X_train["question"].mode()[0], inplace=True)
        df2["span"].fillna(X_train["span"].mode()[0], inplace=True)
        # df2["answer"].fillna(X_train["answer"].mode()[0], inplace=True)
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

    #######################################################
    # print(
    #     Fore.YELLOW + "Check missing values in categorical variables in X_train:\n",
    #     X_train[categorical].isnull().sum(),
    # )
    # print(
    #     Fore.YELLOW + "Check missing values in categorical variables in X_train:\n",
    #     X_train[categorical].isnull().sum(),
    # )
    # print(Fore.YELLOW + "Check missing values in X_train:\n", X_train.isnull().sum())
    # print(Fore.YELLOW + "Check missing values in X_test\n", X_test.isnull().sum())

    ######################################################
    # encode remaining variables with one-hot encoding
    encoder = ce.OneHotEncoder(
        cols=[
            "question",
            "span",
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
    # print("Results are:", y_pred)

    ######################################################
    # Model Accuracy
    print(
        Fore.LIGHTBLUE_EX
        + "\nModel accuracy score: {0:0.4f}%".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

    ######################################################
    # Compare the train-set and test-set accuracy
    # y_pred_train = gnb.predict(X_train)
    # # print(
    # #     "Training-set accuracy score: {0:0.4f}%".format(
    # #         accuracy_score(y_train, y_pred_train) * 100
    # #     )
    # # )

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


def main():
    # split_train_test()
    get_features()


main()
end_time = time.time()
print(Fore.WHITE + "\nExecution Time = ", (end_time - start_time), "seconds")
