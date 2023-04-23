import math
import pandas as pd
import numpy as np
# documents
doc1 = "I want to start learning to charge something in life"
doc2 = "reading something about life no one else knows"
doc3 = "Never stop learning"
# query string
query = "life learning"

# term -frequenvy :word occurences in a document


def compute_tf(docs_list):
    for doc in docs_list:
        doc1_lst = doc.split(" ")
        wordDict_1 = dict.fromkeys(set(doc1_lst), 0)

        for token in doc1_lst:
            wordDict_1[token] += 1

        df = pd.DataFrame([wordDict_1])
        idx = 0
        new_col = ["Term Frequency"]
        df.insert(loc=idx, column='Document', value=new_col)
        # print(df)

        print(wordDict_1)


compute_tf([doc1, doc2, doc3])
