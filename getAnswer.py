import pandas as pd

data = pd.read_csv("CSV-Files/TFIDF_CS_dev1.csv")
data_question = pd.read_csv("CSV-Files/QuestionsList.csv")


data_question_list = data_question.loc[data_question['TitleNo'] == 0]
question_list = data['question'].tolist()

question_index = 0
questions = []
while question_index < len(question_list):
    questions.append(question_list[question_index])
    question_index += 54


data = {
    "question": questions,
    "paragNo": data_question_list['ParagraphNo'],
    "titleNo": data_question_list['TitleNo']
}


df = pd.DataFrame(data)
df.to_csv('CSV-Files/Question_Answer1.csv', encoding='utf-8', index=False)

#######################################################################################
#######################################################################################
#######################################################################################

max_data = pd.read_csv("CSV-Files/MAX_CS_dev1.csv")
actual_data = pd.read_csv("CSV-Files/Question_Answer1.csv")

hit_list = []
actualParag_list = []
inListIndex_list = []
paragNo_list = actual_data['paragNo']
for question in questions:
    tempData = max_data.loc[max_data['question'] == question].copy()
    parag_list = tempData['paragraph_no'].tolist()
    current_data = actual_data.loc[actual_data['question'] == question].copy()

    if parag_list.count(paragNo_list[current_data.index[0]]) != 0:
        hit_list.append(1)
        actualParag_list.append(paragNo_list[current_data.index[0]])
        inListIndex_list.append(parag_list.index(
            paragNo_list[current_data.index[0]]))

    else:
        hit_list.append(0)
        actualParag_list.append(paragNo_list[current_data.index[0]])
        inListIndex_list.append(paragNo_list[current_data.index[0]])

new_data = {
    "question": questions,
    "hit_miss": hit_list,
    "index_in_list": inListIndex_list,
    "actual_parag": actualParag_list,
    "title_no": actual_data['titleNo'],
}

df = pd.DataFrame(new_data)
df.to_csv('CSV-Files/Actual_Answer1.csv', encoding='utf-8', index=False)
