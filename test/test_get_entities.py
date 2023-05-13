from service.get_entities import get_doc_entities


def test_get_doc_entities():
    file = "doc"
    entity_type = "person"
    result = get_doc_entities(file, entity_type)
    assert len(result) > 0
