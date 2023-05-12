import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.utils import fetch_archive_from_http, print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever, TransformersSummarizer
from haystack.pipelines import ExtractiveQAPipeline, SearchSummarizationPipeline
from pprint import pprint

from haystack.utils import convert_files_to_docs
from haystack.nodes import PDFToTextConverter, TextConverter, PreProcessor
from haystack.pipelines import Pipeline

# https://www.cnbc.com/2023/05/12/ukraine-war-live-updates-latest-news-on-russia-and-the-war-in-ukraine.html
doc_dir = "data/web_docs"
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]

text_converter = TextConverter(
    valid_languages=["en"]
)
# all_docs = convert_files_to_docs(dir_path=doc_dir)


# This is a default usage of the PreProcessor.
# Here, it performs cleaning of consecutive whitespaces
# and splits a single large document into smaller documents.
# Each document is up to 1000 words long and document breaks cannot fall in the middle of sentences
# Note how the single document passed into the document gets split into 5 smaller documents
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=150,
    split_respect_sentence_boundary=True,
)
# docs_default = preprocessor.process(all_docs)
# print(f"n_docs_input: 1\nn_docs_output: {len(docs_default)}")

# build doc store
document_store = InMemoryDocumentStore(use_bm25=True)

# build index
indexing_pipeline = TextIndexingPipeline(document_store, text_converter, preprocessor)
indexing_pipeline.run_batch(file_paths=files_to_index)

# build retriever
retriever = BM25Retriever(document_store=document_store)

# build the summarizer
summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-cnn-12-6")

# build pipeline
pipe = SearchSummarizationPipeline(summarizer=summarizer, retriever=retriever, generate_single_summary=True)


def print_answer(preds, question):
    answer_text = preds['documents'][0].meta['summary']
    print(f"\nQuestion: {question}\n\nAnswer: {answer_text}\n-----\n")
    return


# run samples

question = "Who is the Ukranian president?"
prediction = pipe.run(
    query=question,
    params={
        "Retriever": {"top_k": 10}
    }
)
print_answer(prediction, question)

question = "What did the Russian Defense Ministry say?"
prediction = pipe.run(
    query=question,
    params={
        "Retriever": {"top_k": 10}
    }
)
print_answer(prediction, question)
