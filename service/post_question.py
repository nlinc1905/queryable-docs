import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, TransformersSummarizer
from haystack.pipelines import SearchSummarizationPipeline
from haystack.nodes import TextConverter, PreProcessor

from .custom_haystack_retreiver import CustomEmbeddingRetriever


DOC_DIR = "data/web_docs"
FILES_TO_INDEX = [DOC_DIR + "/" + f for f in os.listdir(DOC_DIR)]
USE_BM25 = False


class QuestionAnswer:
    def __init__(self):
        self.pipe = self.setup()

    @staticmethod
    def setup():
        """
        Sets ups and returns a Haystack pipeline for summarization based document search.
        """
        # set up text converter
        text_converter = TextConverter(
            valid_languages=["en"]
        )

        # set up pre-preprocessor
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="word",
            split_length=150,
            split_respect_sentence_boundary=True,
        )

        # build doc store
        document_store = InMemoryDocumentStore(embedding_dim=384, use_bm25=USE_BM25)

        # build index
        indexing_pipeline = TextIndexingPipeline(document_store, text_converter, preprocessor)
        indexing_pipeline.run_batch(file_paths=FILES_TO_INDEX)

        # build retriever
        if USE_BM25:
            retriever = BM25Retriever(document_store=document_store)
        else:
            retriever = CustomEmbeddingRetriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
            document_store.update_embeddings(retriever)

        # build the summarizer
        summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-cnn-12-6")

        # build pipeline
        pipe = SearchSummarizationPipeline(summarizer=summarizer, retriever=retriever, generate_single_summary=True)

        return pipe

    def get_answer(self, question:str) -> str:
        """
        Retrieves the answer to the provided question, using the documents stored in the pipeline's index.

        :param question: The question you want to search for an answer to
        """
        # get predictions (answers)
        predictions = self.pipe.run(
            query=question,
            params={
                "Retriever": {"top_k": 10}
            }
        )
        pred = predictions['documents'][0].meta['summary']

        return pred
