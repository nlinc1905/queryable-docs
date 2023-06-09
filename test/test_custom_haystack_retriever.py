import unittest
from haystack.schema import Document

from service.custom_haystack_retreiver import CustomEmbeddingRetriever


class CustomEmbeddingRetrieverTestCase(unittest.TestCase):

    def setUp(self):
        self.docs = [Document(content="test_1"), Document(content="test_2")]
        self.queries = [Document(content="test_1"), Document(content="test_2")]
        self.retriever = CustomEmbeddingRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_embed_documents(self):
        result = self.retriever.embed_documents(self.docs)
        assert result.shape == (2, 384)

    def test_embed_queries(self):
        result = self.retriever.embed_documents(self.queries)
        assert result.shape == (2, 384)
