import pandas as pd
import spacy
import json
import nltk

nlp = spacy.load("en_core_web_sm")
# data = pd.read_csv("Features/question_answer_dev1.csv")
with open("CSV-Files/devSplit/dev1.json") as f:
    data = json.load(f)


def find_answer_sentence(paragraph_no, answer):
    contexts = data["data"][0]["paragraphs"][paragraph_no]["context"]
    sentences = nltk.sent_tokenize(contexts)
    result = []
    for sentence in sentences:
        if answer in sentence:
            result.append(sentence)

    # print(result)
    return result


find_answer_sentence(
    0,
    "Denver Broncos",
)
# find_answer_sentence(
#     0, "Denver Broncos", "Which NFL team represented the AFC at Super Bowl 50?"
# )

############################################

import re
import networkx as nx
from practnlptools.tools import Annotator

annotator = Annotator()
text = """Robots in popular culture are there to remind us of the awesomeness of unbound human agency."""
dep_parse = annotator.getAnnotations(text, dep_parse=True)["dep_parse"]

dp_list = dep_parse.split("\n")
pattern = re.compile(r".+?\((.+?), (.+?)\)")
edges = []
for dep in dp_list:
    m = pattern.search(dep)
    edges.append((m.group(1), m.group(2)))

graph = nx.Graph(edges)  # Well that was easy

print(nx.shortest_path_length(graph, source="Robots-1", target="awesomeness-12"))
