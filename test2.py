import networkx as nx
import spacy

nlp = spacy.load("en_core_web_sm")
document = nlp(
    "Which NFL team represented the AFC at Super Bowl 50?",
)

print("document: {0}".format(document))

# Load spacy's dependency tree into a networkx graph
edges = []
for token in document:
    for child in token.children:
        edges.append(
            (
                "{0}-{1}".format(token.lower_, token.i),
                "{0}-{1}".format(child.lower_, child.i),
            )
        )

graph = nx.Graph(edges)
print(graph.edges)

# print(nx.shortest_path_length(graph, source="robots-0", target="awesomeness-11"))
# print(nx.shortest_path(graph, source="afc-5", target="which-0"))
print(nx.shortest_path(graph, source="which-0", target="afc-5"))
# print(nx.shortest_path(graph, source="robots-0", target="agency-15"))
