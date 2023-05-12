import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.utils import fetch_archive_from_http, print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint


# initialize doc store
document_store = InMemoryDocumentStore(use_bm25=True)


# fetch GoT data for testing
doc_dir = "data/build_your_first_question_answering_system"
fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir
)

# build index
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

# build retriever
retriever = BM25Retriever(document_store=document_store)

# build reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# build pipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# run sample
prediction = pipe.run(
    query="Who is the father of Arya Stark?",
    params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    }
)
# pprint(prediction)
print_answers(
    prediction,
    details="minimum" ## Choose from `minimum`, `medium`, and `all`
)