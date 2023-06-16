# import spacy
# from spacy import displacy
# nlp = spacy.load("en_core_web_sm")
# from practnlptools.tools import Annotator
# import spacy
# import networkx as nx
# span = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season."
# # doc = nlp(span)
# # print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")
# # for token in doc:
# #     print("{:<15} {:^10} {:>15}".format(
# #         str(token.head.text), str(token.dep_), str(token.text)))
# # displacy.render(doc, style='dep')
# annotator = Annotator()
# annotator.getAnnotations(span)

######################
# import spacy
# from spacy import displacy

# nlp = spacy.load("en_core_web_sm")
# # sentence = "The quick brown fox jumping over the lazy dog"
# sentence = "Robots in popular culture are there to remind us of the awesomeness of unbounded human agency."
# doc = nlp(sentence)
# print(f"{'Node (from)-->':<15} {'Relation':^10} {'-->Node (to)':>15}\n")
# for token in doc:
#     print(
#         "{:<15} {:^10} {:>15}".format(
#             str(token.head.text), str(token.dep_), str(token.text)
#         )
#     )
# displacy.render(doc, style="dep")


# token.dep_ → shows dependency (label)
# token.head.text → shows head
# token.head.text → shows dependent
# token.i → shows position

######################

# import spacy
# py_text = "spacy dependency parser in python"
# py_nlp = spacy.load ("en_core_web_sm")
# py_doc = py_nlp (py_text)
# for token in py_doc:
#             print (token.py_text, token.dep_,"token head is", token.head.py_text,
# [child for child in token.children]

######################


def dependency_parser(span):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(span)

    position_label = []
    children = []

    # for token in doc:
    #     print(token.text, "\t", token.dep_, "\t", spacy.explain(token.dep_))

    # to explore the dependency parse structure
    # for token in doc:
    #     print(token.text)
    #     ancestors = [t.text for t in token.ancestors]  # chilrder attributes
    #     print(ancestors)
    # print("########################")

    # to see all the children token
    for token in doc:
        token_childs = []
        # print(token.text)
        position_label.append(
            (
                str(token.dep_),
                str(token.head.text) + " " + str(token.head.i),
                str(token.text) + " " + str(token.i),
            )
        )
        childs = [str(t.text) + " " + str(t.i) for t in token.children]
        token_childs = [str(token) + " " + str(token.i), childs]
        # print(position_label)
        # print(children)
        children.append(token_childs)
    # print(len(children))
    # print("########################")

    # see the subtree that the token is in
    # for token in doc:
    #     print(token.text)
    #     subtree = [t.text for t in token.subtree]
    #     print(subtree)
    # print("########################")

    root = ""
    for token in doc:
        if token.dep_ == "ROOT":
            root = str(token) + " " + str(token.i)
            # print("ROOt = ", str(token) + " " + str(token.i))
            break

    return (root, position_label, children)


# sentence = "I have seldom heard him mention her under any other name."
# sentence = "Deemed universities charge huge fees."
# sentence = "I prefer the morning flight through Denver."
# sentence = "Robots in popular culture are there to remind us of the awesomeness of unbounded human agency((."
# DP = dependency_parser(sentence)
# root = DP[0]
# position_label = DP[1]
# children = DP[2]
