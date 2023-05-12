import spacy


NER = spacy.load("en_core_web_sm")
DOC_DIR = "data/web_docs"


def get_doc_entities(file: str, entity_type: str = "person"):
    file = DOC_DIR + "/" + str(file) + ".txt"
    with open(file, "r") as f:
        doc = NER(f.read())
        ents = {
            word.text: word.sent
            for word in doc.ents if word.label_.lower() == entity_type.lower()
        }
    return list(ents)
