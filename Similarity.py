import spacy
import datetime
import pandas as pd
from importjson import document_context, document_question, questions1

print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
print("Reading File..")

nlp = spacy.load("en_core_web_md")

questions_list = []
parag_sim_doc_titles_val = []
parag_sim_doc_parag_no = []
parag_sim_doc_title_no = []

print("Check similarity for each question and paragraphs..")
for title_index in range(len(document_context)):
    print("title number:", title_index)
    for question in document_question[title_index]:
        print("question number:", document_question[title_index].index(
            question), " out of:", len(document_question[title_index]))
        for paragraph in document_context[title_index]:
            questions_list.append(question)
            parag_sim_doc_title_no.append(title_index)
            parag_sim_doc_parag_no.append(
                document_context[title_index].index(paragraph))

            q = nlp(question)
            p = nlp(paragraph)
            val = q.similarity(p)
            parag_sim_doc_titles_val.append(val)

print("Creating dataframe..")
data = {
    "question": questions_list,
    "similarity": parag_sim_doc_titles_val,
    "parag_no": parag_sim_doc_parag_no,
    "title_no": parag_sim_doc_title_no,
}
df = pd.DataFrame(data)
df.to_csv('Similarity_dev1.csv', encoding='utf-8', index=False)

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
