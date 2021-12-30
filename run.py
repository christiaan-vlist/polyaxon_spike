import argparse

# Polyaxon
from polyaxon import tracking

from model import train

if __name__ == "__main__":

    # Polyaxon
    tracking.init()

    # Train and eval the model with given parameters.
    # Polyaxon
    model_path = "model.joblib"
    metrics = train(model_path=model_path)

    # Logging metrics to Polyaxon
    print("Testing metrics: {}", metrics)

    # Polyaxon
    tracking.log_metrics(**metrics)

    # Logging the model
    tracking.log_model(
        model_path, name="useritem-model", versioned=False
    )
