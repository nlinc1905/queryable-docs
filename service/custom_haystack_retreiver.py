import typing as t
import numpy as np
import json
import requests
from requests.adapters import HTTPAdapter, Retry
from haystack.schema import Document
from haystack.nodes.retriever import EmbeddingRetriever


EMBED_ENDPOINT = "http://34.111.150.97/embed"
EMBED_BATCH_ENDPOINT = "http://34.111.150.97/embed_batch"


def call_endpoint(endpoint: str, data):
    """Makes post request to the given endpoint with the provided data."""

    # make request, wait and retry if API is overloaded
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[404, 504]
    )
    session.mount(endpoint, HTTPAdapter(max_retries=retries))
    resp = session.post(endpoint, json=data)

    # raise error if retries fail
    resp.raise_for_status()

    # parse response
    ans = resp.json()
    return ans


class CustomEmbeddingRetriever(EmbeddingRetriever):
    def embed_documents(self, documents: t.List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings, one per input document, shape: (docs, embedding_dim)
        """
        documents = {"texts": [doc.content for doc in documents]}
        embeds = call_endpoint(EMBED_BATCH_ENDPOINT, documents)
        embeds = np.asarray(json.loads(embeds))
        return embeds

    def embed_queries(self, queries: t.List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        # for backward compatibility: cast pure str input
        if isinstance(queries, str):
            queries = [queries]
        assert isinstance(queries, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        embeds = call_endpoint(EMBED_BATCH_ENDPOINT, queries)
        embeds = np.asarray(json.loads(embeds))
        return embeds
