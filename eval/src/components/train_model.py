from typing import NamedTuple
from kfp.v2 import dsl


# google/cloud-sdk:slim base image has gsutil and Python3.9
# see: https://github.com/GoogleCloudPlatform/cloud-sdk-docker#components-installed-in-each-tag
@dsl.component(
    base_image="google/cloud-sdk:slim",
    output_component_file="pipeline_specs/train_model.yaml",
    packages_to_install=["sentence-transformers==2.2.2"]
)
def train_model(model_name: str, model_path: str, gcs_bucket: str) -> NamedTuple(
    "Outputs",
    [
        ("dummy", str),
    ],
):
    import logging
    import torch
    from sentence_transformers import SentenceTransformer
    from subprocess import call

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

    def save_model_to_gcs(model_to_save, local_path: str, bucket_folder_uri: str) -> None:
        """
        Saves the model locally, and then uses gsutil to copy the local directory to a gcs
        bucket folder.
        """
        # save the model locally first
        model_to_save.save(local_path)

        # copy files to bucket
        call(["gsutil", "cp", "-r", f"./${local_path}/*", f"${bucket_folder_uri}"], shell=True)

        # remove local directory
        call(["rm", "-R", f"${local_path}"], shell=True)

    logger = get_logger()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Using device {device}")

    gcs_model_folder = f"{model_name.lower()}"
    gcs_model_folder_uri = f"gs://{gcs_bucket}/{gcs_model_folder}"

    # placeholder comment for future training

    # save the model to gcs
    save_model_to_gcs(
        model_to_save=model,
        local_path=model_path,
        bucket_folder_uri=gcs_model_folder_uri
    )

    # To make sure downstream tasks are dependent on this one finishing,
    # before they can begin, we will return a dummy string that downstream
    # tasks will take as input.
    dummy = "done"
    return (dummy,)
