from typing import NamedTuple
from kfp.v2 import dsl


@dsl.component(
    base_image="python:3.9-slim",
    output_component_file="pipeline_specs/evaluate_model.yaml",
    packages_to_install=[
        "google-cloud-storage==2.9.0",
        "sentence-transformers==2.2.2",  # includes torch
        "scikit-learn==0.24.2",  # includes numpy and scipy
        "pandas==1.3.4",
    ]
)
def evaluate_model(
        model_name: str,
        local_model_path: str,
        local_data_folder: str,
        gcs_bucket: str,
        gcs_data_folder: str,
        ann_index_type: str,
        nbr_queries: int,
        nbr_documents: int,
        train_model_output: str,
        metrics: dsl.Output[dsl.Metrics],
) -> NamedTuple(
    "Outputs",
    [
        ("deploy_decision", str),
    ],
):
    import logging
    import torch
    import json
    import typing as t
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from google.cloud import storage
    from pathlib import Path
    from scipy.spatial.distance import cdist
    from sklearn.metrics import ndcg_score, label_ranking_average_precision_score, roc_auc_score

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

    def download_files_from_cloud_storage(
            gcs_client,
            logger,
            bucket: str,
            bucket_folder: str,
            local_folder: str,
            exclude_list: t.List[str] = None
    ) -> None:
        """Copies files from cloud storage folder to local folder."""
        bucket = gcs_client.get_bucket(bucket)
        blobs = bucket.list_blobs(prefix=bucket_folder)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            path_split = blob.name.split("/")
            filename = path_split[-1]
            directory = "/".join(path_split[:-1])
            if exclude_list is not None and (directory in exclude_list or filename in exclude_list):
                logger.info(f"Skipping artifact {blob.name}")
                next
            path = Path(f"{local_folder}/{directory}")
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {blob.name} to: {path}/{filename}\n")
            blob.download_to_filename(f"{path}/{filename}")

    def create_embeddings(model, model_embed_dim: int, queries: dict, corpus: dict) -> t.Tuple[np.array, np.array]:
        """Creates embeddings for the query strings and texts"""
        q_embeddings = np.array(
            model.encode(list(queries.values())),
            dtype=np.float32
        ).reshape(-1, model_embed_dim)

        docs = [doc['text'] for doc in corpus.values()]
        d_embeddings = np.array(
            model.encode(docs),
            dtype=np.float32
        ).reshape(-1, model_embed_dim)

        return q_embeddings, d_embeddings

    def rescale(d: float, dampening_factor: float = 1):
        """
        Inverts distance to be similarity.  Dividing by d + 1 ensures that 0 becomes 1 in the new scale.

        :param d: distance measurement
        :param dampening_factor: Increase this to flatten the curve and make larger values fall off slower
            in the new scale.  This parameter should be tuned to what makes sense.  1.5 is a reasonable
            starting value, see: https://www.desmos.com/calculator/eluirxagoz
        """
        return (1 / (d + 1)) ** (1 / dampening_factor)

    def score_relevance(query_vector: np.array, doc_embeddings: np.array, text_ids: list = None):
        """Computes scaled similarity (relevance) to a query vector for every document."""
        dist_to_query = cdist(query_vector.reshape(1, -1), doc_embeddings, metric='euclidean')
        dist_to_query = rescale(d=dist_to_query[0])
        if text_ids is not None:
            output = {text_ids[i]: dist_to_query[i] for i in range(len(text_ids))}
        else:
            output = {i: dist_to_query[i] for i in range(len(dist_to_query))}
        return output

    def combine_labels_and_preds(qrels: dict, qrels_pred: dict, queries: dict, docs: dict) -> pd.DataFrame:
        # assemble the labeled data
        dfs = []
        for query, relations in qrels.items():
            doc_ids = list(relations)
            labels = list(relations.values())
            q_name = [query] * len(doc_ids)
            q_text = queries[query]
            doc_texts = [docs[did]['text'] for did in doc_ids]
            qdf = pd.DataFrame({
                "query": q_name,
                "query_text": q_text,
                "doc_id": doc_ids,
                "doc_text": doc_texts,
                "label": labels,
            })
            qdf = qdf.sort_values(["label", "doc_id"], ascending=False).reset_index(drop=True)
            qdf['binary_label'] = np.where(qdf['label'] > 0, 1, 0)
            dfs.append(qdf)
        df = pd.concat(dfs, axis=0)

        # assemble the predictions
        pred_dfs = []
        for query, relations in qrels_pred.items():
            doc_ids = list(relations)
            scores = list(relations.values())
            q_name = [query] * len(doc_ids)
            pdf = pd.DataFrame({"query": q_name, "doc_id": doc_ids, "similarity_score": scores})
            pred_dfs.append(pdf)
        pred_df = pd.concat(pred_dfs, axis=0)

        # combine the predictions with the labels
        final_df = pd.merge(left=df, right=pred_df, how='left', on=['query', 'doc_id'])
        final_df['similarity_score'].fillna(0, inplace=True)
        final_df.sort_values(["query", "label", "similarity_score"], ascending=False, inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        return final_df

    def score_results(df: pd.DataFrame):
        """
        Split data into groups, score each query group's ranking by similarity score,
        and re-combine.
        """
        groups = [y for x, y in df.groupby('query')]
        for group in groups:
            group['ndcg_top10'] = ndcg_score([group['label']], [group['similarity_score']], k=10)
            group['ndcg_top100'] = ndcg_score([group['label']], [group['similarity_score']], k=100)
            group['ndcg'] = ndcg_score([group['label']], [group['similarity_score']])
            group['lrap'] = label_ranking_average_precision_score(
                [group['binary_label']], [group['similarity_score']]
            )
            group['roc_auc'] = roc_auc_score(group['binary_label'], group['similarity_score'])
        return pd.concat(groups, axis=0)

    def save_dataframe_to_cloud_storage(
            gcs_client,
            df: pd.DataFrame,
            bucket_name: str,
            folder_path: str,
            fname: str
    ) -> None:
        """Uploads dataframe as CSV file to Google Cloud Storage bucket."""
        bucket = gcs_client.bucket(bucket_name)
        bucket.blob(f'{folder_path}/{fname}.csv').upload_from_string(df.to_csv(), 'text/csv')

    logger = get_logger()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Using device {device}")

    gcs_model_folder = f"{model_name.lower()}"
    gcs_client = storage.Client()

    # initializing the model from local download fails for a bug,
    # see: https://github.com/UKPLab/sentence-transformers/issues/1590
    # so we will cheat (since we know we have not trained the model) and load directly from HF hub
    # when the bug is fixed, uncomment the code below to load from local

    # download the trained model from gcs to the container
    # download_files_from_cloud_storage(
    #     gcs_client=gcs_client,
    #     logger=logger,
    #     bucket=gcs_bucket,
    #     bucket_folder=gcs_model_folder,
    #     local_folder=local_model_path,
    #     exclude_list=["training_and_eval_artifacts"]
    # )

    # load the model from local path
    # model = SentenceTransformer(f"{local_model_path}/{gcs_model_folder}", device=device)
    model = SentenceTransformer(model_name, device=device)

    # download the eval data from gcs to the container
    download_files_from_cloud_storage(
        gcs_client=gcs_client,
        logger=logger,
        bucket=gcs_bucket,
        bucket_folder=gcs_data_folder,
        local_folder=local_data_folder
    )

    # load the eval data from local path
    with open(f"{local_data_folder}/{gcs_data_folder}/corpus.json", "r") as file:
        corpus = json.load(file)
    with open(f"{local_data_folder}/{gcs_data_folder}/queries.json", "r") as file:
        queries = json.load(file)
    with open(f"{local_data_folder}/{gcs_data_folder}/qrels.json", "r") as file:
        qrels = json.load(file)

    # embed the queries and the documents
    all_query_embed, all_doc_embed = create_embeddings(
        model=model,
        model_embed_dim=384,
        queries=queries,
        corpus=corpus
    )

    # map document IDs from the corpus to their index values in the corpus
    doc_id_to_tid_map = {k: v for v, k in enumerate(list(corpus))}
    tid_to_doc_id_map = dict(enumerate(list(corpus)))

    # map query IDs to their embeddings
    query_id_to_embed_map = {qid: all_query_embed[i, :] for i, qid in enumerate(list(queries))}

    # map document IDs to their embeddings
    doc_id_to_embedding_map = {did: all_doc_embed[i, :] for i, did in enumerate(list(corpus))}

    # iterate over the query-doc relations and score them for relevancy
    qrels_pred = {}
    for query_id, doc_ids in qrels.items():
        # get the query embedding
        query_embed = query_id_to_embed_map[query_id].reshape(1, -1)

        # get the document embeddings for the docs found in this query's relations
        text_ids = [doc_id_to_tid_map[did] for did in list(doc_ids)]
        doc_embed = np.concatenate([doc_id_to_embedding_map[did] for did in list(doc_ids)], axis=0).reshape(-1, 384)

        # score the documents and map scores back to original doc IDs
        query_doc_scores = score_relevance(query_vector=query_embed, doc_embeddings=doc_embed, text_ids=text_ids)
        query_doc_scores = {tid_to_doc_id_map[k]: v for k, v in query_doc_scores.items()}

        # store results for this query
        q_res = {query_id: query_doc_scores}
        qrels_pred.update(q_res)

    df_pred = combine_labels_and_preds(
        qrels=qrels,
        qrels_pred=qrels_pred,
        queries=queries,
        docs=corpus
    )

    # make sure every query:doc pair has been accounted for
    assert (len(df_pred) == sum([len(v) for v in qrels.values()]))
    # score relevancy predictions against labels and check query:doc pairs again, as score_results produces a new df
    df = score_results(df=df_pred)
    assert (len(df) == sum([len(v) for v in qrels.values()]))

    # metrics to be logged
    metrics.log_metric("average_ndcg_top10", df["ndcg_top10"].mean())
    metrics.log_metric("average_ndcg_top100", df["ndcg_top100"].mean())
    metrics.log_metric("average_ndcg", df["ndcg"].mean())
    metrics.log_metric("average_lrap", df["lrap"].mean())
    metrics.log_metric("average_roc_auc", df["roc_auc"].mean())

    # saving eval data with preds should be an optional step, but it could be useful for analysis in BigQuery
    save_dataframe_to_cloud_storage(
        gcs_client=gcs_client,
        df=df,
        bucket_name=gcs_bucket,
        folder_path=gcs_data_folder,
        fname=f"eval_{model_name}.csv"
    )

    # if the model performs well enough, it should be deployed
    deploy_decision = "true"
    return (deploy_decision,)
