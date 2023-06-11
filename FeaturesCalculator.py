
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")


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


def get_root_matching(question, span):
    # question = ' '.join(question)
    # span = ' '.join(span)

    q_doc = nlp(question)
    s_doc = nlp(span)
    question_dep = ''
    span_dep = ''

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
            tf = (occurance/N)
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
            tf = (occurance/len(span))
            result += tf

    return result

def categoricalAnswers(answersList):
    cardinals=[]
    person =[]
    date=[]
    loc =[]
    money =[]
    gpe =[]
    org =[]
    event =[]
    other = []

    NER = spacy.load("en_core_web_sm")
    for answer in answersList:
        text2= NER(answer)
        for word in text2.ents:
            if (word.label_ == "CARDINAL"):
                cardinals.append(word.text)
            elif(word.label_ == "PERSON" ):
                person.append(word.text)
            elif(word.label_ == "DATE"):
                date.append(word.text)
            elif(word.label_ == "LOC"):
                loc.append(word.text)
            elif(word.label_ == "MONEY"):
                money.append(word.text)
            elif(word.label_ == "ORG"):
                org.append(word.text)
            elif(word.label_ == "GPE"):
                gpe.append(word.text)
            elif(word.label_ == "EVENT"):
                event.append(word.text)
            else:
                other.append(word.text)


def get_uni_bigram_overlap(question, span):
    pass


def get_unlexicalized_path(question, span):
    pass


def get_constituency_parse(spans):
    pass


def get_features(question, span):
    # question = stopword_func(question)
    # span = stopword_func(span)

    # lexicalized_feature =
    matching_word_frequency = get_matching_word_frequency(question, span)
    # matching_biagram_frequency =
    root_match = get_root_matching(question, span)
    length = len(span)
    span_word_frequency = get_span_TFIDF(span)
    # bm25 = get_BM25()
    # unigram_bigram_overlap =
    # consistant_label =
    # span_POS_tags =
    # dependency_tree_path =

    print(matching_word_frequency)

span = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season."
question = "Which NFL team represented the AFC at Super Bowl 50?"
get_features(question, span)
