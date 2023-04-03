import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from importjson import questions1,answers1,contexts1



token_data = []


def stopword_func():
    with open('dev.json') as f:
        data = json.load(f)

    # # example_sent = """This is a sample sentence,showing off the stop words filtration."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(('and','I','A','And','So','.',',',')','By','(',"''",'Other','The',';','however', 'still','the','They','For','also','In','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','This'))

    str1 = ""
    for ele in answers1:
        str1 +=ele
    str2 = ""
    for ele in questions1:
        str2 +=ele
    str3 = ""
    for ele in contexts1:
        str3+=ele
    # print(str3)

    word_tokens1 = word_tokenize(str1)
    filtered_sentence = [w for w in word_tokens1 if not w.lower() in stop_words]
    filtered_sentence = []
    for w in word_tokens1:
        if w not in stop_words:
            filtered_sentence.append(w)

    word_tokens2 = word_tokenize(str2)
    filtered_sentence2 = [w for w in word_tokens2 if not w.lower() in stop_words]
    filtered_sentence2 = []
    for w in word_tokens2:
        if w not in stop_words:
            filtered_sentence2.append(w)

    word_tokens3 = word_tokenize(str3)
    filtered_sentence3 = [w for w in word_tokens3 if not w.lower() in stop_words]
    filtered_sentence3 = []
    for w in word_tokens3:
        if w not in stop_words:
            filtered_sentence3.append(w)
    # print(word_tokens)

    # print(filtered_sentence)

    # print(filtered_sentence2)

    # print(filtered_sentence3)
    token_data.append(filtered_sentence)
    token_data.append(filtered_sentence2)
    token_data.append(filtered_sentence3)
    # print(token_data)
    return token_data

stopword_func()