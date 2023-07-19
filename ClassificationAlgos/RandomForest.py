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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
from sklearn.ensemble import RandomForestClassifier

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

df = pd.read_csv("ClassificationAlgos/Features_dev1.csv", encoding="utf8")

########################################################
col_names = df.columns
# print(Fore.YELLOW + "Columns:", col_names)
# print(
#     "\nCheck distribution of target_class column:\n",
#     df["answer"].value_counts(),
# )
# print(
#     "\nView the percentage distribution of target_class column:\n",
#     df["answer"].value_counts() / np.float(len(df)),
# )
# print("\nCheck for missing values in variables:\n", df.isnull().sum())

########################################################
X = df.drop(["answer"], axis="columns")
y = df["answer"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

########################################################
# impute missing categorical variables with most frequent value
for df2 in [X_train, X_test]:
    df2["question"].fillna(X_train["question"].mode()[0], inplace=True)
    df2["span"].fillna(X_train["span"].mode()[0], inplace=True)
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
    df2["trigram_overlap"].fillna(X_train["trigram_overlap"].mode()[0], inplace=True)
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
    df2["hamming_distance"].fillna(X_train["hamming_distance"].mode()[0], inplace=True)
    df2["jaccard_distance"].fillna(X_train["jaccard_distance"].mode()[0], inplace=True)
    df2["edit_distance"].fillna(X_train["edit_distance"].mode()[0], inplace=True)
    df2["span_length"].fillna(X_train["span_length"].mode()[0], inplace=True)
    df2["question_length"].fillna(X_train["question_length"].mode()[0], inplace=True)

######################################################
# Feature Scaling
# encode remaining variables with one-hot encoding
encoder = ce.OrdinalEncoder(
    cols=[
        "question",
        "span",
    ]
)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

######################################################
# instantiate the classifier
rfc = RandomForestClassifier(random_state=0)
# fit the model
rfc.fit(X_train, y_train)
# Predict the Test set results
y_pred = rfc.predict(X_test)
# Check accuracy score
print(
    Fore.LIGHTBLUE_EX
    + "Model accuracy score with 10 decision-trees : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "\nRF : Model accuracy score with 10 decision-trees : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

##############################################################################################################
# instantiate the classifier with n_estimators = 100
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
# fit the model to the training set
rfc_100.fit(X_train, y_train)
# Predict on the test set results
y_pred_100 = rfc_100.predict(X_test)
# Check accuracy score
print(
    "Model accuracy score with 100 decision-trees : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred_100) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "RF : Model accuracy score with 100 decision-trees : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )
##############################################################################################################
# Build Random Forest model on selected features
X = df.drop(
    [
        "answer",
        "paragNo",
        "titleNo",
        "trigram_overlap",
        "span_length",
        "question_length",
        "root_matching",
    ],
    axis="columns",
)
y = df["answer"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)
########################################################
# impute missing categorical variables with most frequent value
for df2 in [X_train, X_test]:
    df2["question"].fillna(X_train["question"].mode()[0], inplace=True)
    df2["span"].fillna(X_train["span"].mode()[0], inplace=True)
    df2["wh_word"].fillna(X_train["wh_word"].mode()[0], inplace=True)
    df2["syntactic_divergence"].fillna(
        X_train["syntactic_divergence"].mode()[0], inplace=True
    )
    # df2["root_matching"].fillna(X_train["root_matching"].mode()[0], inplace=True)
    df2["span_TFIDF"].fillna(X_train["span_TFIDF"].mode()[0], inplace=True)
    df2["matching_word_frequency"].fillna(
        X_train["matching_word_frequency"].mode()[0], inplace=True
    )
    df2["bigram_overlap"].fillna(X_train["bigram_overlap"].mode()[0], inplace=True)
    # df2["trigram_overlap"].fillna(X_train["trigram_overlap"].mode()[0], inplace=True)
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
    df2["hamming_distance"].fillna(X_train["hamming_distance"].mode()[0], inplace=True)
    df2["jaccard_distance"].fillna(X_train["jaccard_distance"].mode()[0], inplace=True)
    df2["edit_distance"].fillna(X_train["edit_distance"].mode()[0], inplace=True)
    # df2["span_length"].fillna(X_train["span_length"].mode()[0], inplace=True)
    # df2["question_length"].fillna(X_train["question_length"].mode()[0], inplace=True)

########################################################

# encode categorical variables with ordinal encoding
encoder = ce.OrdinalEncoder(
    cols=[
        "question",
        "span",
    ]
)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# instantiate the classifier with n_estimators = 100
clf = RandomForestClassifier(random_state=0)
# fit the model to the training set
clf.fit(X_train, y_train)
# Predict on the test set results
y_pred = clf.predict(X_test)
# Check accuracy score
print(
    Fore.MAGENTA
    + "\nModel accuracy score with some variable removed : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "RF : Model accuracy score with some variable removed : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )
##############################################################################################################

end_time = time.time()
print(Fore.WHITE + "\nExecution Time = ", (end_time - start_time), "seconds")
