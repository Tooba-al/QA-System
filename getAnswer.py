import pandas as pd

data = pd.read_csv("CSV-Files/TFIDF_CS_dev2.csv")
data_question = pd.read_csv("CSV-Files/QuestionsList.csv")


data_question_list = data_question.loc[data_question['TitleNo'] == 1]
question_list = data['question'].tolist()

question_index = 0
questions = []
# question_data = pd.read_csv("CSV-Files/questions2.csv")
# questions = question_data['question']
while question_index < len(question_list):
    questions.append(question_list[question_index])
    question_index += 49

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
        inListIndex_list.append('-')

new_data = {
    "question": questions,
    "hit_miss": hit_list,
    "index_in_list": inListIndex_list,
    "actual_parag": actualParag_list,
    "title_no": actual_data['titleNo'],
}

df = pd.DataFrame(new_data)
df.to_csv('CSV-Files/Actual_Answer2.csv', encoding='utf-8', index=False)

#######################################################################################
#######################################################################################
#######################################################################################

hit_count = (hit_list.count(1)/len(hit_list))*100
first_choice_count = 0
second_choice_count = 0
third_choice_count = 0
fourth_choice_count = 0
fifth_choice_count = 0
for hit in hit_list:
    if hit == 1:
        index = hit_list.index(hit)

        if (inListIndex_list[index] == 0):
            first_choice_count += 1
        elif (inListIndex_list[index] == 1):
            second_choice_count += 1
        elif (inListIndex_list[index] == 2):
            third_choice_count += 1
        elif (inListIndex_list[index] == 3):
            fourth_choice_count += 1
        elif (inListIndex_list[index] == 4):
            fifth_choice_count += 1


not_number = inListIndex_list.count('-')
if (len(inListIndex_list)-not_number) == 0:
    choice1 = (first_choice_count/len(inListIndex_list))*100
    choice2 = (second_choice_count/len(inListIndex_list))*100
    choice2 = (second_choice_count/len(inListIndex_list))*100
    choice3 = (third_choice_count/len(inListIndex_list))*100
    choice4 = (fourth_choice_count/len(inListIndex_list))*100
    choice5 = (fifth_choice_count/len(inListIndex_list))*100
else:
    choice1 = (first_choice_count/(len(inListIndex_list)-not_number))*100
    choice2 = (second_choice_count/(len(inListIndex_list)-not_number))*100
    choice3 = (third_choice_count/(len(inListIndex_list)-not_number))*100
    choice4 = (fourth_choice_count/(len(inListIndex_list)-not_number))*100
    choice5 = (fifth_choice_count/(len(inListIndex_list)-not_number))*100
accu_data = {
    "hit%": [hit_count],
    "first_choice%": [choice1],
    "second_choice%": [choice2],
    "third_choice%": [choice3],
    "fourth_choice%": [choice4],
    "fifth_choice%": [choice5],
}

df = pd.DataFrame(accu_data)
df.to_csv('CSV-Files/Result_Actual_Answer2.csv', encoding='utf-8', index=False)
