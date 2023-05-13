from fastapi import FastAPI, Depends, Request
from fastapi_pagination import Params
from pydantic import BaseModel

from .post_question import QuestionAnswer
from .get_entities import get_doc_entities


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str


app = FastAPI()
question_answer = QuestionAnswer()


@app.post("/question")
def answer_question(data: QuestionRequest) -> QuestionResponse:
    answer_resp = question_answer.get_answer(question=data.question)
    return QuestionResponse(
        answer=answer_resp.strip().replace("\n", "")
    )


@app.get("/people/{doc_name}")
def show_people(doc_name: str, request: Request, params: Params = Depends()):
    # entities = get_doc_entities(file=doc_name)
    entities = get_doc_entities(file="doc")
    return entities
