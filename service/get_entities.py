import spacy
from typing import List


NER = spacy.load("en_core_web_sm")
DOC_DIR = "data/web_docs"


def get_doc_entities(file: str, entity_type: str = "person") -> List[dict]:
    """
    Retrieves a list of entities of the given entity type, from the given file.

    :param file: The file you want to scan for entities
    :param entity_type: The entity type from spacy that you want to focus on
    """
    file = DOC_DIR + "/" + str(file) + ".txt"
    with open(file, "r") as f:
        doc = NER(f.read())
        ents = {
            word.text: word.sent
            for word in doc.ents if word.label_.lower() == entity_type.lower()
        }
    return list(ents)
