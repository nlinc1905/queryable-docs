from fastapi import FastAPI, Depends, Request
from fastapi_pagination import Params
from pydantic import BaseModel
from typing import List

from .post_question import get_answer
from .get_entities import get_doc_entities


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str


app = FastAPI()


@app.post("/question")
def answer_question(data: QuestionRequest) -> QuestionResponse:
    answer_resp = get_answer(question=data.question)
    return QuestionResponse(
        answer=answer_resp.strip().replace("\n", "")
    )


@app.get("/people/{doc_name}")
def show_people(doc_name: str, request: Request, params: Params = Depends()):
    # entities = get_doc_entities(file=doc_name)
    entities = get_doc_entities(file="doc")
    return entities
