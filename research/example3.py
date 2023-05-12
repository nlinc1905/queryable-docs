import spacy
import os


ner = spacy.load("en_core_web_sm")

doc_dir = "data/web_docs"
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]


def get_doc_entities(file, entity_type):
    with open(file, "r") as f:
        doc = ner(f.read())
        ents = {word.text: {"type": word.label_, "sentence": word.sent} for word in doc.ents if word.label_.lower() == entity_type.lower()}
    return ents


entities = get_doc_entities(files_to_index[0], entity_type="person")
for n, p in entities.items():
    print(f"\nName: {n}\n\nSentence where person appears:\n\n{p['sentence']}\n\n-----\n")
print(list(entities))
