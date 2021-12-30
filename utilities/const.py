"""A collection of (semi-)constants which are used throughout the application.

Functions:
    get_container_image_uri: Return the URI for the model image repository
    get_dataframe_dtypes: The data types for the resulting pandas dataframe
    get_engine_service_account: The service account used by the engine
    get_google_project: Name of the GCP project used for deploying & storing
    get_local_pipeline_folder: Path to the folder for storing models locally
    get_location: The location for running the Vertex AI models
    get_remote_pipeline_folder: Path to the folder for storing models remotely
    set_google_project: Set the name of the GCP project used
"""
from __future__ import annotations

import pathlib
import typing

import cloudpathlib
import google.cloud.storage as storage
import numpy
import pandas


#: The Google Cloud Platform project to run remote operations in
_GOOGLE_PROJECT: str = ""


def get_container_image_uri() -> cloudpathlib.AnyPath:
    """Return the URI of the container which stores all model images."""
    return pathlib.Path(
        f"{get_location()}-docker.pkg.dev/{get_google_project()}/engine-image-repository/recommendation-engine"
    )


def get_dataframe_dtypes() -> dict[str, typing.Any]:
    """Return the dataframe data types in which to cast data."""
    return {
        "timestamp": pandas.DatetimeTZDtype("ns", "Europe/Amsterdam"),
        "user": pandas.CategoricalDtype(),
        "item": pandas.CategoricalDtype(),
        "weight": numpy.float64,
    }


def get_engine_service_account() -> str:
    """Return the service account for the recommendations engine."""
    return f"rec-engine-sa@{get_google_project()}.iam.gserviceaccount.com"


def get_google_project() -> str:
    """Get the currently set google project.

    Usefull, because Vertex AI doesn't play nice with auto-detection.

    Reference:
        https://cloud.google.com/vertex-ai/docs/training/code-requirements
    """
    return _GOOGLE_PROJECT


def set_google_project(project_id: str) -> None:
    """Set the google cloud project to use throughout the code.

    Usefull, because VertexAI doesn't play nice with auto-detection.

    Args:
        project_id: The ID of the new google cloud project to use

    Raises:
        TypeError: Thrown if the project is set more than once

    Reference:
        https://cloud.google.com/vertex-ai/docs/training/code-requirements
    """
    global _GOOGLE_PROJECT
    assert _GOOGLE_PROJECT == "", TypeError("The Google project can only be set once!")
    _GOOGLE_PROJECT = project_id

    # Initialize the CloudPathlib library.
    client = storage.Client(project=project_id)
    cloudpathlib.GSClient(storage_client=client).set_as_default_client()


def get_local_pipeline_folder() -> cloudpathlib.AnyPath:
    """Return the local folder in which trained pipelines can be stored."""
    return pathlib.Path("./trained_pipelines")


def get_location() -> str:
    """Return the location in which all Google operations should occur."""
    return "europe-west4"


def get_remote_pipeline_folder() -> cloudpathlib.AnyPath:
    """Return the remote folder in which trained pipelines can be stored."""
    project = get_google_project()
    return cloudpathlib.CloudPath(f"gs://re-{project}/trained_pipelines")


def get_model_state_topic() -> str:
    """Return the remote folder in which trained pipelines can be stored."""
    topic_name = "projects/{project_id}/topics/{topic}".format(
        project_id=get_google_project(),
        topic="rec-engine-model-status",
    )
    return topic_name
