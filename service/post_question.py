import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, TransformersSummarizer
from haystack.pipelines import SearchSummarizationPipeline
from haystack.nodes import PDFToTextConverter, TextConverter, PreProcessor
from haystack.pipelines import Pipeline


DOC_DIR = "data/web_docs"
FILES_TO_INDEX = [DOC_DIR + "/" + f for f in os.listdir(DOC_DIR)]


def setup():
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
    document_store = InMemoryDocumentStore(use_bm25=True)

    # build index
    indexing_pipeline = TextIndexingPipeline(document_store, text_converter, preprocessor)
    indexing_pipeline.run_batch(file_paths=FILES_TO_INDEX)

    # build retriever
    retriever = BM25Retriever(document_store=document_store)

    # build the summarizer
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-cnn-12-6")

    # build pipeline
    pipe = SearchSummarizationPipeline(summarizer=summarizer, retriever=retriever, generate_single_summary=True)

    return pipe


def get_answer(question:str) -> str:
    # get predictions (answers)
    predictions = PIPE.run(
        query=question,
        params={
            "Retriever": {"top_k": 10}
        }
    )
    pred = predictions['documents'][0].meta['summary']

    return pred


PIPE = setup()
