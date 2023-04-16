
from importjson import document_context, document_question, questions1
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import time


start_time = time.time()


questions = []
index_dict_question = {}  # Dictionary to store index for each word
doc_contain_word = []
tfidf_word_paragraph_index = []


def stopword_func(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(('?', 'and', 'I', 'A', 'And', 'So', '.', 'as', 'As', '\'\'', 'could', '[', ']', ',', ')', '\'s', 'By', '(', "''",
                       'Other', '``', ':', '\'', '#', '\'v', 'The', ';', 'however', 'still',
                      'the', 'They', 'For', 'also', 'In', 'This', 'When', 'It', 'many', 'Many', 'so', 'cant', 'Yes', 'yes', 'No', 'no',
                       'These', 'these', 'This', 'Where', 'Which', 'Why', 'How', 'What', 'If', 'Who', 'When'))

    str1 = ""
    for ele in text:
        str1 += ele
    lemmatizer = WordNetLemmatizer()
    word_tokens1 = word_tokenize(str1)
    filtered_sentence = [lemmatizer.lemmatize(
        word) for word in word_tokens1 if not word in set(stop_words)]

    return filtered_sentence


document_sentences = []
all_sentences = []
all_questions = []
doc_word_set2 = []
ques_word_set2 = []

# stop_word_context
# Preprocessing the text data
print("Run Stop-Word function for documents..")
for title in document_context:
    for document in title:
        word_tokens = stopword_func(document)
        document_sentences.append(word_tokens)

        for word in word_tokens:
            if word not in doc_word_set2:
                doc_word_set2.append(word)

    all_sentences.append(document_sentences)
    # all_sentences : total_list -> title -> paragraph

# stop_word_question
# Preprocessing the text data
print("Run Stop-Word function for questions..")
question_title_index = []

# for each title in each document
for title_question in document_question:

    # for each question in the same title
    for question in title_question:
        word_tokens = stopword_func(question)
        questions.append(word_tokens)

        for word in word_tokens:
            question_title_index.append(
                document_question.index(title_question))

            if word not in ques_word_set2:
                ques_word_set2.append(word)

    all_questions.append(questions)

# Tokens are belong to context words  => set make array of list

# Set of vocab
doc_word_set = set(doc_word_set2)
ques_word_set = set(ques_word_set2)

# Total documents in our corpus
print("number of sentence: ", len(all_sentences))
print("number of question: ", len(all_questions))

print("Creating an index for each word in each question..")
index_dict_question = {}  # Dictionary to store index for each word
i = 0
for word in ques_word_set:
    index_dict_question[word] = i
    i += 1


# Term Frequency
def termfreq(paragraph, word):
    tf_result = 0
    # for document in paragraph:
    occurance = 0
    N = len(paragraph)
    occurance = paragraph.count(word)

    if occurance != 0:
        tf_result = (occurance/N)

    return tf_result


# Inverse Document Frequency
def inverse_doc_freq(word, paragraph_list):
    # if doc_contain_word != []:
    #     idf = np.log((len(paragraph)) / len(doc_contain_word))
    # else:
    count = 0
    N = len(paragraph_list)     # total number of documents
    for paragraph in paragraph_list:
        if paragraph.count(word) > 0:
            count += 1
            # print(paragraph.count(word))
    if count != 0:
        idf = np.log(N+1 / count)
    else:
        idf = 0
    return idf


question_word_w2 = []
question_word_q2 = []
question_title_index2 = []
for title_question_index in range(len(all_questions)):
    for question in all_questions[title_question_index]:
        for word in question:
            for paragraph in title:
                tfidf_word_paragraph_index.append(title.index(paragraph))
                question_word_w2.append(word)
                question_word_q2.append(question)
                question_title_index2.append(title_question_index)


def tf_idf(title, question, tf_idf_vec, index_dict):
    # create Sparse matrix
    for word in question:
        for paragraph in title:
            tf = termfreq(paragraph, word)
            idf = inverse_doc_freq(word, title)
            value = tf*idf
            tf_idf_vec[index_dict[word]] = value

    return tf_idf_vec


# TF-IDF Encoded text corpus
print("for each title..")
print("Use the TF/IDF function..")

# print(index_dict_question)

tf_idf_vec = np.zeros((len(tfidf_word_paragraph_index),))
for title_question_index in range(len(all_questions)):
    for question in all_questions[title_question_index]:
        vec = tf_idf(all_sentences[title_question_index],
                     question,
                     tf_idf_vec,
                     index_dict_question)

print("Number of all words in all questions: ", len(question_word_w2))

print("Creating a model for all data..")
data = {
    'question': question_word_q2,
    'word': question_word_w2,
    'TF_IDF': vec,
    'paragraph_no': tfidf_word_paragraph_index,
    'title_no': question_title_index2,
}

df = pd.DataFrame(data)
df.to_csv('outputs/output_dev1.csv', encoding='utf-8', index=False)


end_time = time.time()
print("execution time -> ", (end_time-start_time)/60, " minutes")

print('End of "optimal_tf_idf.py"')
