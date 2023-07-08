import json
import time


start_time = time.time()

document_contexts = []
document_questions = []
document_answers = []
document_answer_spans = []
document_file_names = []
document_file_no = []


# read json file
def readFile():
    with open("bAbI_DS/qa1_single-supporting-fact_train.txt") as f:
        lines = f.readlines()

        for line in lines:
            line = line.split()[1:]
            question_flag = 0

            for word in line:
                if word.count("?") != 0:
                    question_flag = 1
                    document_questions.append(line[:-2])
                    document_answers.append(line[-2])
                    break
            if question_flag == 0:
                document_contexts.append(line)

        # print(document_contexts)
        # print(document_questions)
        # print(document_answers)


readFile()
