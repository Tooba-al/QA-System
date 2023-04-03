
from importjson import document_context, document_question, questions1
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import time


start_time = time.time()


# indexes : begin_f
# begin_f = 0
all_sentences = []
all_questions = []
doc_word_set2 = []
ques_word_set2 = []
title_document_index = []
sent_doc_tfidf = []
ques_doc_tfidf = []
questions = []
document_sentences = []
# index_dict_context = {} #Dictionary to store index for each word
index_dict_question = {}  # Dictionary to store index for each word
doc_contain_word = []
all_words = []
question_word_q = []
question_word_w = []
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


# stop_word_context
# #Preprocessing the text data
print("Run Stop-Word function for documents..")
for title in document_context:
    for document in title:
        word_tokens = stopword_func(document)
        document_sentences.append(word_tokens)

        for word in word_tokens:
            if word not in doc_word_set2:
                doc_word_set2.append(word)
                sent_doc_tfidf.append(title)

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
            question_word_q.append(question)
            question_word_w.append(word)
            question_title_index.append(
                document_question.index(title_question))

            if word not in ques_word_set2:
                ques_word_set2.append(word)
                ques_doc_tfidf.append(question)

    all_questions.append(questions)

# Tokens are belong to context words  => set make array of list

# Set of vocab
doc_word_set = set(doc_word_set2)
ques_word_set = set(ques_word_set2)

# Total documents in our corpus
print("number of sentence: ", len(all_sentences))
print("number of question: ", len(all_questions))

# #Creating an index for each word in our vocab.
# index_dict_context = {} #Dictionary to store index for each word
# i = 0
# for word in doc_word_set:
#     index_dict_context[word] = i
#     i += 1

print("Creating an index for each word in each question..")
index_dict_question = {}  # Dictionary to store index for each word
i = 0
for word in ques_word_set:
    index_dict_question[word] = i
    i += 1


# # Create a count dictionary
# def count_dict(sentences, word_set):
#     word_count = {}
#     for word in word_set:
#         word_count[word] = 0

#         for sent in sentences:
#             if word in sent:
#                 word_count[word] += 1

#     return word_count


# print("Create a count dictionary for document...")
# for title in all_sentences:
#     word_count_doc = count_dict(title, doc_word_set)

# print("Create a count dictionary for all questions..")
# for title in all_questions:
#     word_count_ques = count_dict(title, doc_word_set)


# Term Frequency
def termfreq(paragraph, word):
    tf_list = []
    tf_result = 1
    for document in paragraph:
        occurance = 0
        N = len(document)

        # for token in document:
        #     if token == word:
        #         occurance += 1
        occurance = document.count(word)

        if occurance != 0:
            # doc_contain_word.append(title.index(document))
            tf_result *= (occurance/N)

    # result = 1
    # for tf in tf_list:
    #     result *= tf

    return tf_result


# Inverse Document Frequency
def inverse_doc_freq(paragraph):
    if doc_contain_word != []:
        idf = np.log(len(paragraph)+1 / len(doc_contain_word))
    else:
        idf = np.log(len(paragraph)+1 / 1)

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


def tf_idf(title, question, all_words, index_dict):
    # create Sparse matrix
    tf_idf_vec = np.zeros((len(tfidf_word_paragraph_index),))

    for word in question:
        for paragraph in title:
            tf = termfreq(paragraph, word)
            idf = inverse_doc_freq(paragraph)
            value = tf*idf
            tf_idf_vec[index_dict[word]] = value

    return tf_idf_vec


# TF-IDF Encoded text corpus
print("for each title..")
print("Use the TF/IDF function..")

for title_question_index in range(len(all_questions)):
    for question in all_questions[title_question_index]:
        #   tf_idf (title,
        #           question,
        #           all_question_word,
        #           index_dic)

        vec = tf_idf(all_sentences[title_question_index],
                     question,
                     question_word_w2,
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
df.to_csv('outputs/output_dev48.csv', encoding='utf-8', index=False)
# df.to_hdf('outputs/output_sampleH.h5', encoding='utf-8',
#           index=False, mode='w', key='df')


end_time = time.time()
print("execution time -> ", (end_time-start_time)/60, " minutes")

print('End of "optimal_tf_idf.py"')
