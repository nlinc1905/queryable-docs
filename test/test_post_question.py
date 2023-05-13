import unittest
from haystack.pipelines import SearchSummarizationPipeline

from service.post_question import QuestionAnswer


class QuestionAnswerTestCase(unittest.TestCase):

    def setUp(self):
        self.question = "string"
        self.question_answer = QuestionAnswer()

    def test_setup(self):
        result = self.question_answer.setup()
        assert isinstance(result, SearchSummarizationPipeline)

    def test_get_answer(self):
        result = self.question_answer.get_answer(question=self.question)
        assert len(result) > 0
