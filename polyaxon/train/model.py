"""The code for a variety of training commands in the CLI.

Functions:
    train_local: Run a small training job on the local machine
    train_remote: Start a full-on training job in Vertex AI
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import joblib
from ..data_loaders.mock_loader import MockProfileLoader
from ..top_n_models.DemoUserEpisodes import DemoUserEpisodes


def train(
    max_train_timestamp: datetime.datetime | None,
    hyperparameters: dict[str, Any],
    model_path: str
) -> None:
    # Start keeping track of the running time
    start_dt = datetime.datetime.now()

    # Create a new data loader
    data_loader = MockProfileLoader(max_datetime=max_train_timestamp)

    m = DemoUserEpisodes(data_loader=data_loader, hyperparameters=hyperparameters.copy())
    m.fit()
    model = m.to_trained()

    # Stop the clock
    end_dt = datetime.datetime.now()

    # Print the results
    logging.info("--- report ---")
    logging.info(f"model written to : {model_path}")

    if model_path:
        joblib.dump(model, model_path)

    results = {
        "started": start_dt,
        "ended": end_dt,
        "elapsed": end_dt - start_dt,
    }
    return results