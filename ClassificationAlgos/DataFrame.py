import json
import time
import nltk
import pandas as pd


start_time = time.time()
contexts1 = []
questions1 = []
answers1 = []

document_context = []
document_question = []
document_answer = []


# read json file
def readFile():
    with open("CSV-Files/devSplit/dev2.json") as f:
        # with open('dev.json') as f:
        data = json.load(f)

    contexts = []
    spans = []
    questions = []
    answers = []
    titleNos = []
    paragraphNos = []
    # extract contexts1, questions1 and answers1
    for i in range(len(data["data"])):
        # contexts1 = []
        # questions1 = []
        # answers1 = []

        for j in range(len(data["data"][i]["paragraphs"])):
            # contexts1.append(data["data"][i]["paragraphs"][j]["context"])
            context = data["data"][i]["paragraphs"][j]["context"]
            sentences = nltk.sent_tokenize(context)

            for span in sentences:
                for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                    question = data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                    questions.append(question)
                    answer = data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0][
                        "text"
                    ]
                    answers.append(answer)
                    spans.append(span)
                    contexts.append(context)
                    titleNos.append(2)
                    paragraphNos.append(j)

    data = {
        "context": contexts,
        "span": spans,
        "question": questions,
        "answer": answers,
        "titleNo": titleNos,
        "paragraphNo": paragraphNos,
    }
    df = pd.DataFrame(data)
    df.to_csv("ClassificationAlgos/DF_dev2.csv", encoding="utf-8", index=False)

    end_time = time.time()
    print("execution time -> ", (end_time - start_time), " seconds")
    print('\nEnd of "Create Data Frame.py\n')


readFile()
