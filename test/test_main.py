from fastapi.testclient import TestClient

from service.main import app


client = TestClient(app)


def test_answer_question():
    question = "string"
    response = client.post("/question", json={
        "question": question
    })
    assert response.status_code == 200


def test_show_people():
    doc_name = "string"
    response = client.get(f"/people/{doc_name}")
    assert response.status_code == 200
