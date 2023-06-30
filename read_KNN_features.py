from KNN_features import *
import pandas as pd
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


train_data = pd.read_csv("KNN/TrainData_dev1.csv")
question_list = train_data["question"].tolist()
answer_list = train_data["answer"].tolist()
span_list = train_data["span"].tolist()
paragNo_list = train_data["paragraphNo"].tolist()
titleNo_list = train_data["titleNo"].tolist()

# print(len(question_list))
dict_datas = []
for index in range(len(question_list)):
    this_result = main(
        question_list[index],
        span_list[index],
        answer_list[index],
        titleNo_list[index],
        paragNo_list[index],
    )

    dict_datas.append(this_result)
    print(
        Fore.GREEN
        + "\nEnd of question "
        + str(index)
        + "/"
        + str(len(question_list))
        + "...\n"
    )


csv_file = "KNN/KNN_Features_CSV_dev111.csv"
csv_columns = [
    "paragNo",
    "titleNo",
    "question",
    "span",
    "answer",
    "syntatic_divergence",
    "root_matching",
    "span_TFIDF",
    "matching_word_frequency",
    "bigram_overlap",
    "trigram_overlap",
    "span_word_frequency",
    "bigram_TFIDF",
    "trigram_TFIDF",
    "minkowski_distance",
    "manhattan_distance",
    "euclidean_distance",
    "hamming_distance",
    "jaccard_distance",
    "edit_distance",
    "consistant_labels",
    "span_POS_tags",
    "span_length",
    "question_length",
]
try:
    with open(csv_file, "w") as features_file:
        writer = csv.DictWriter(features_file, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_datas:
            writer.writerow(data)
except IOError:
    print(Fore.RED + "I/O error")


end_time = time.time()
print(Fore.RED + "Execution Time = ", (end_time - start_time) / 60)
