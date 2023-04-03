import json
import time


start_time = time.time()
contexts1 = []
questions1 = []
answers1 = []

document_context = []
document_question = []
document_answer = []


# read json file
def readFile():
    # with open('dev2.json') as f:
    with open('devSplit/dev48.json') as f:
        data = json.load(f)

        # extract contexts1, questions1 and answers1
        for i in range(len(data['data'])):
            contexts1 = []
            questions1 = []
            answers1 = []

            for j in range(len(data['data'][i]['paragraphs'])):
                contexts1.append(data['data'][i]['paragraphs'][j]['context'])

                for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                    questions1.append(
                        data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                    answers1.append(
                        data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])

            document_context.append(contexts1)
            document_question.append(questions1)
            document_answer.append(answers1)

        # count the word count for each element in the list and replace it with the len

        print("number of titles: ", len(document_context))
        print("for each title: ")
        print("\tnumber of contexts: ", len(contexts1))
        print("\tnumber of questions: ", len(questions1))
        print("\tnumber of answers: ", len(answers1))

        end_time = time.time()
        print("execution time -> ", (end_time-start_time), " seconds")
        print('\nEnd of "importjason.py\n')

        return data


readFile()
