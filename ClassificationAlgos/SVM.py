import spacy
import pandas as pd
from nltk.stem import WordNetLemmatizer
import datetime
from colorama import Fore
import time
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.metrics import accuracy_score
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
# print("\nDraw boxplots to visualize outliers:")
# plt.figure(figsize=(24, 20))


# plt.subplot(4, 2, 1)
# fig = df.boxplot(column="edit_distance")
# fig.set_title("")
# fig.set_ylabel("edit_distance")


# plt.subplot(4, 2, 2)
# fig = df.boxplot(column="jaccard_distance")
# fig.set_title("")
# fig.set_ylabel("jaccard_distance")


# plt.subplot(4, 2, 3)
# fig = df.boxplot(column="hamming_distance")
# fig.set_title("")
# fig.set_ylabel("hamming_distance")


# plt.subplot(4, 2, 4)
# fig = df.boxplot(column="euclidean_distance")
# fig.set_title("")
# fig.set_ylabel("euclidean_distance")


# plt.subplot(4, 2, 5)
# fig = df.boxplot(column="manhattan_distance")
# fig.set_title("")
# fig.set_ylabel("manhattan_distance")


# plt.subplot(4, 2, 6)
# fig = df.boxplot(column="minkowski_distance")
# fig.set_title("")
# fig.set_ylabel("minkowski_distance")


# plt.subplot(4, 2, 7)
# fig = df.boxplot(column="span_TFIDF")
# fig.set_title("")
# fig.set_ylabel("span_TFIDF")


# plt.subplot(4, 2, 8)
# fig = df.boxplot(column="bigram_TFIDF")
# fig.set_title("")
# fig.set_ylabel("bigram_TFIDF")

########################################################
X = df.drop(["answer"], axis="columns")
y = df["answer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
encoder = ce.OneHotEncoder(
    cols=[
        "question",
        "span",
    ]
)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

######################################################
cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

######################################################
# instantiate classifier with default hyperparameters
svc = SVC()
# fit classifier to training set
svc.fit(X_train, y_train)

# make predictions on test set
y_pred = svc.predict(X_test)

######################################################
######################################################
######################################################
# Run SVM with polynomial kernel
# compute and print accuracy score
print(
    Fore.LIGHTBLUE_EX
    + "\nModel accuracy score with default hyperparameters: {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "\nSVM : Model accuracy score with default hyperparameters: {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

######################################################
# instantiate classifier with rbf kernel and C=100
svc = SVC(C=100.0)
# fit classifier to training set
svc.fit(X_train, y_train)
# make predictions on test set
y_pred = svc.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

######################################################
# instantiate classifier with rbf kernel and C=1000
svc = SVC(C=1000.0)
# fit classifier to training set
svc.fit(X_train, y_train)
# make predictions on test set
y_pred = svc.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

######################################################
# instantiate classifier with linear kernel and C=1.0
linear_svc = SVC(kernel="linear", C=1.0)
# fit classifier to training set
linear_svc.fit(X_train, y_train)
# make predictions on test set
y_pred_test = linear_svc.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with linear kernel and C=1.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred_test) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with linear kernel and C=1.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )
######################################################
# instantiate classifier with linear kernel and C=100.0
linear_svc100 = SVC(kernel="linear", C=100.0)
# fit classifier to training set
linear_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred = linear_svc100.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with linear kernel and C=100.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with linear kernel and C=100.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

######################################################
# instantiate classifier with linear kernel and C=1000.0
linear_svc1000 = SVC(kernel="linear", C=1000.0)
# fit classifier to training set
linear_svc1000.fit(X_train, y_train)
# make predictions on test set
y_pred = linear_svc1000.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )
######################################################
# Compare the train-set and test-set accuracy
y_pred_train = linear_svc.predict(X_train)
print(
    "Training-set accuracy score: {0:0.4f}%".format(
        accuracy_score(y_train, y_pred_train) * 100
    )
)

######################################################
# Check for overfitting and underfitting
print("Training set score: {:.4f}%".format(linear_svc.score(X_train, y_train) * 100))
print("Test set score: {:.4f}%".format(linear_svc.score(X_test, y_test) * 100))

######################################################
######################################################
######################################################
# Run SVM with polynomial kernel
# instantiate classifier with polynomial kernel and C=1.0
poly_svc = SVC(kernel="poly", C=1.0)
# fit classifier to training set
poly_svc.fit(X_train, y_train)
# make predictions on test set
y_pred = poly_svc.predict(X_test)
# compute and print accuracy score
print(
    Fore.LIGHTGREEN_EX
    + "\nModel accuracy score with polynomial kernel and C=1.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

#######################################################
# instantiate classifier with polynomial kernel and C=100.0
poly_svc100 = SVC(kernel="poly", C=100.0)
# fit classifier to training set
poly_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred = poly_svc100.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with polynomial kernel and C=100.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with polynomial kernel and C=100.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

#######################################################
# instantiate classifier with polynomial kernel and C=100.0
poly_svc1000 = SVC(kernel="poly", C=1000.0)
# fit classifier to training set
poly_svc1000.fit(X_train, y_train)
# make predictions on test set
y_pred = poly_svc1000.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with polynomial kernel and C=1000.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with polynomial kernel and C=1000.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

########################################################
########################################################
########################################################
# Run SVM with sigmoid kernel and C=1.0
# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc = SVC(kernel="sigmoid", C=1.0)
# fit classifier to training set
sigmoid_svc.fit(X_train, y_train)
# make predictions on test set
y_pred = sigmoid_svc.predict(X_test)
# compute and print accuracy score
print(
    Fore.MAGENTA
    + "\nModel accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

########################################################
# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100 = SVC(kernel="sigmoid", C=100.0)
# fit classifier to training set
sigmoid_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred = sigmoid_svc100.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )

########################################################
# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc1000 = SVC(kernel="sigmoid", C=1000.0)
# fit classifier to training set
sigmoid_svc1000.fit(X_train, y_train)
# make predictions on test set
y_pred = sigmoid_svc1000.predict(X_test)
# compute and print accuracy score
print(
    "Model accuracy score with sigmoid kernel and C=1000.0 : {0:0.4f}%".format(
        accuracy_score(y_test, y_pred) * 100
    )
)
with open("ClassificationAlgos/results.txt", "a") as text_file:
    text_file.write(
        "SVM : Model accuracy score with sigmoid kernel and C=1000.0 : {0:0.4f}%\n".format(
            accuracy_score(y_test, y_pred) * 100
        )
    )
