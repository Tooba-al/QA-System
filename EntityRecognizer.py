import spacy
import datetime
import pandas as pd
from importjson import document_context

print("Start Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
print("Pipeline component for named entity recognition..")
named_entities_texts = []
named_entities_labels = []
contexts_list = []
titles_list = []
contexts_list = []
nlp = spacy.load("en_core_web_sm")

print("For each title, extract tokens and their lables..")

for title in document_context:
    for parag in title:
        doc = nlp(parag)
        ents = doc.ents
        for ent in ents:
            named_entities_labels.append(ent.label_)
            named_entities_texts.append(ent.text)
            contexts_list.append(title.index(parag))
            titles_list.append(document_context.index(title))

data = {
    "word": named_entities_texts,
    "label": named_entities_labels,
    "parag_no": contexts_list,
    "title_no": titles_list,
}
df = pd.DataFrame(data)
df.to_csv('NE_dev.csv', encoding='utf-8', index=False)

print("End Time: = %s:%s:%s" % (datetime.datetime.now().hour,
      datetime.datetime.now().minute, datetime.datetime.now().second))
