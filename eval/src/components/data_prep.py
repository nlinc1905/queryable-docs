from typing import NamedTuple
from kfp.v2 import dsl


@dsl.component(
    base_image="python:3.9-slim",
    output_component_file="pipeline_specs/data_prep.yaml",
    packages_to_install=["google-cloud-storage", "tqdm", "more-itertools"]
)
def data_prep(data_path: str, bucket: str, bucket_folder: str) -> NamedTuple(
    "Outputs",
    [
        ("nbr_queries", int),
        ("nbr_documents", int),
    ],
):
    import logging
    import os
    import json
    import csv
    import urllib.request
    import typing as t
    from more_itertools import take
    from zipfile import ZipFile
    from tqdm import tqdm
    from google.cloud import storage

    def get_logger():
        """Sets up logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        return logger

    def download_and_unzip_dbpedia(save_path: str, dataset: str = "dbpedia-entity") -> None:
        """
        Downloads and unzips the DBpedia dataset.
        DBpedia documentation: https://github.com/iai-group/DBpedia-Entity/
        Download link comes from: https://github.com/beir-cellar/beir
        """
        if not os.path.exists(f"{save_path}/dbpedia-entity/corpus.jsonl"):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            fname = "dbpedia-entity.zip"

            # download the model to the current working directory
            urllib.request.urlretrieve(url, fname)

            # extract to a new folder
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with ZipFile(fname, "r") as zipf:
                zipf.extractall(path=save_path)

            # remove downloaded zip file
            os.remove(fname)

    def load_dbpedia(from_path: str) -> t.Tuple[
        t.Dict[str, t.Dict[str, str]], t.Dict[str, str], t.Dict[str, t.Dict[str, int]]
    ]:
        """
        Loads the DBpedia dataset.
        Code adapted from: https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py

        The queries dict is a key:value map of the query ID to the query text.  Here is an example of a query:
        queries['TREC_Entity-9']

        The query relations dict maps query IDs to a dict of document IDs and their relevancy scores (0, 1, 2), where
        0 = irrelevant, 1 = relevant, and 2 = highly relevant.  Not all doc IDs appear in the dict results for a given
        query.
        qrels['TREC_Entity-9']

        Corpus maps document IDs to a dict of texts and titles.
        corpus['<dbpedia:Todd_Levy>']
        """
        corpus, queries, qrels = {}, {}, {}

        corpus_file = f"{from_path}/dbpedia-entity/corpus.jsonl"
        queries_file = f"{from_path}/dbpedia-entity/queries.jsonl"
        qrels_file = f"{from_path}/dbpedia-entity/qrels/test.tsv"

        # load the corpus
        num_lines = sum(1 for _ in open(corpus_file, 'rb'))
        with open(corpus_file, encoding='utf8') as file:
            for line in tqdm(file, total=num_lines):
                line = json.loads(line)
                corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

        # load the queries
        with open(queries_file, encoding='utf8') as file:
            for line in file:
                line = json.loads(line)
                queries[line.get("_id")] = line.get("text")

        # load the query:doc relationships
        reader = csv.reader(
            open(qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL
        )
        next(reader)

        for _, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score

        return corpus, queries, qrels

    def upload_to_cloud_storage(data: dict, bucket_name: str, folder_path: str, fname: str) -> None:
        """Uploads dictionary as JSON file to Google Cloud Storage bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(folder_path + "/" + fname)
        blob.upload_from_string(json.dumps(data), timeout=int(60 * 5))
        return

    def validate(logger, queries: dict, qrels: dict, corpus: dict) -> set:
        """Runs data validation"""

        # data validation: how many queries have no relevant documents?
        queries_without_relevant_docs = 0
        for _, relations in qrels.items():
            total_irrelevant = 0
            for _, label in relations.items():
                if label == 0:
                    total_irrelevant += 1
            if total_irrelevant == len(relations):
                queries_without_relevant_docs += 1
        logger.info(
            f"Queries without relevant documents: "
            f"{queries_without_relevant_docs} / {len(qrels)} = "
            f"{round(100 * queries_without_relevant_docs / len(qrels), 2)}%"
        )

        # data validation: how many queries have no query relations?
        q_minus_qrel = len(set(queries) - set(qrels))
        logger.info(
            f"Queries without query relations: "
            f"{q_minus_qrel} / {len(set(queries))} = "
            f"{round(100 * q_minus_qrel / len(set(queries)), 2)}%"
        )

        # data validation: how many query relations have no query data?
        qrel_minus_q = len(set(qrels) - set(queries))
        logger.info(
            f"Query relations without queries: "
            f"{qrel_minus_q} / {len(set(qrels))} = "
            f"{round(100 * qrel_minus_q / len(set(qrels)), 2)}%"
        )

        # data validation: how many documents in query relations have no document data?
        unique_docs_in_qrels = set()
        for _, rels in qrels.items():
            unique_docs_in_qrels = unique_docs_in_qrels.union(set(rels))
        qrel_minus_docs = len(unique_docs_in_qrels - set(corpus))
        logger.info(
            f"Documents in query relations without document data: "
            f"{qrel_minus_docs} / {len(unique_docs_in_qrels)} = "
            f"{round(100 * qrel_minus_docs / len(unique_docs_in_qrels), 2)}%"
        )

        # data validation: how many documents have no query relations?
        docs_minus_qrels = len(set(corpus) - unique_docs_in_qrels)
        logger.info(
            f"Documents without query relations: "
            f"{docs_minus_qrels} / {len(set(corpus))} = "
            f"{round(100 * docs_minus_qrels / len(set(corpus)), 2)}%"
        )

        # assertion should be True if data validation counts are correct
        assert len(set(corpus)) - len(unique_docs_in_qrels) == docs_minus_qrels

        return unique_docs_in_qrels

    def clean_data(
            logger,
            queries: dict,
            qrels: dict,
            corpus: dict,
            unique_docs_in_qrels: set
    ) -> t.Tuple[dict, dict]:
        """Cleans data"""

        # remove documents without query relations - they will be of no use for evaluation
        corpus = {k: v for k, v in corpus.items() if k in unique_docs_in_qrels}
        logger.info(f"New corpus size: {len(corpus)}")

        # remove queries without query relations - they will be of no use for evaluation
        queries = {k: v for k, v in queries.items() if k in set(qrels)}
        logger.info(f"New query count: {len(queries)}")

        return queries, corpus

    def save_data_to_cloud_storage(
            queries: dict, qrels: dict, corpus: dict, gcs_bucket: str, gcs_folder_path: str
    ) -> None:
        """Saves the data to be used for evaluation to Cloud Storage bucket"""
        upload_to_cloud_storage(data=queries, bucket_name=gcs_bucket, folder_path=gcs_folder_path, fname="queries.json")
        upload_to_cloud_storage(data=qrels, bucket_name=gcs_bucket, folder_path=gcs_folder_path, fname="qrels.json")
        upload_to_cloud_storage(data=corpus, bucket_name=gcs_bucket, folder_path=gcs_folder_path, fname="corpus.json")

    logger = get_logger()
    download_and_unzip_dbpedia(save_path=data_path)
    corpus, queries, qrels = load_dbpedia(from_path=data_path)
    unique_docs_in_qrels = validate(logger=logger, queries=queries, qrels=qrels, corpus=corpus)
    queries, corpus = clean_data(
        logger=logger,
        queries=queries,
        qrels=qrels,
        corpus=corpus,
        unique_docs_in_qrels=unique_docs_in_qrels
    )
    save_data_to_cloud_storage(
        queries=queries,
        qrels=qrels,
        corpus=corpus,
        gcs_bucket=bucket,
        gcs_folder_path=bucket_folder
    )

    nbr_queries = len(qrels)
    nbr_documents = len(corpus)
    return (nbr_queries, nbr_documents,)
