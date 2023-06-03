import uuid
from kfp.v2 import dsl, compiler
from google.cloud import aiplatform

from components.data_prep import data_prep
from components.train_model import train_model
from components.evaluate_model import evaluate_model


MODEL_NAME = "all-MiniLM-L6-v2"

PROJECT_ID = "queryable-docs-dev"
REGION = "us-central1"  # select one where running a job on a GPU will not exceed your quota
GCS_BUCKET = "queryable-docs-artifacts-5024"
GCS_PIPELINE_ARTIFACTS_FOLDER = f"{GCS_BUCKET}/{MODEL_NAME.lower()}/training_and_eval_artifacts"
GCS_PIPELINE_ARTIFACTS_URI = f"gs://{GCS_PIPELINE_ARTIFACTS_FOLDER}"

LOCAL_DATA_FOLDER = "data"  # place to store data on container until transferred to cloud storage
LOCAL_MODEL_FOLDER = "local_model"  # place to store model on container until transferred to cloud storage
GCS_DATA_FOLDER = "ir_eval_data"

# type of index to use for approximate nearest neighbor search (ANN)
ANN_INDEX_TYPE = "flat"  # flat is just pairwise L2 distance

# name of the JSON file to be generated with pipeline specs, local path or GCS URI
PIPELINE_SPEC_JSON = "pipeline_specs/train_eval_ir_pipeline.json"
# tie this pipeline run to an experiment
EXPERIMENT_NAME = f"{PROJECT_ID}-ir-{MODEL_NAME.lower()}-{uuid.uuid1()}"


@dsl.pipeline(
    name="train-and-eval-ir-pipeline",
    description="Train and evaluation information retrieval (IR) pipeline.",
    pipeline_root=GCS_PIPELINE_ARTIFACTS_URI
)
def train_eval_ir_pipeline(project: str, ann_index_type: str):
    """
    By default, each component will run on as a Vertex AI CustomJob using an e2-standard-4 machine, with 4 core CPUs
    and 16GB memory.  These configs can be changed, follow the example from
    (https://cloud.google.com/vertex-ai/docs/pipelines/machine-types).
    """
    data_prep_task = data_prep(
        data_path=LOCAL_DATA_FOLDER,
        bucket=GCS_BUCKET,
        bucket_folder=GCS_DATA_FOLDER
    )
    train_model_task = (
        train_model(model_name=MODEL_NAME, model_path=LOCAL_MODEL_FOLDER, gcs_bucket=GCS_BUCKET)
        # .add_node_selector_constraint('cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_T4')
        # .set_gpu_limit(1)
    )
    evaluate_model_task = (
        evaluate_model(
            model_name=MODEL_NAME,
            local_model_path=LOCAL_MODEL_FOLDER,
            local_data_folder=LOCAL_DATA_FOLDER,
            gcs_bucket=GCS_BUCKET,
            gcs_data_folder=GCS_DATA_FOLDER,
            ann_index_type=ann_index_type,
            nbr_queries=data_prep_task.outputs["nbr_queries"],
            nbr_documents=data_prep_task.outputs["nbr_documents"],
            train_model_output=train_model_task.outputs["dummy"],
        )
        # .add_node_selector_constraint('cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_T4')
        # .set_gpu_limit(1)
    )


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=train_eval_ir_pipeline, package_path=PIPELINE_SPEC_JSON)

    ml_pipeline_job = aiplatform.PipelineJob(
        display_name=f"train-and-eval-ir-pipeline:{EXPERIMENT_NAME}",
        template_path=PIPELINE_SPEC_JSON,  # defines the DAG, must match package_path of the compiler
        pipeline_root=GCS_PIPELINE_ARTIFACTS_URI,
        # parameter_values = pipeline arguments, or parameters for the experiment
        parameter_values={"project": PROJECT_ID, "ann_index_type": ANN_INDEX_TYPE},
        enable_caching=True,
        project=PROJECT_ID,
        location=REGION
    )

    ml_pipeline_job.submit(experiment=EXPERIMENT_NAME)
