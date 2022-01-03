from typing import Dict

import joblib
import numpy as np
from flask import Flask, jsonify, make_response, request

import abstract_top_n_model
import DemoUserEpisodes

def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


app = Flask(__name__)
ranker = load_model("./model.joblib")


def get_features(request_data: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            request_data["from_ids"],
            request_data["n"],
        ],
        ndmin=2,
    )


def predict(features: np.ndarray) -> Dict:
    print(features)
    return ranker.model.predict(features[0], features[1])


@app.route("/api/v1/predict", methods=["POST"])
def get_prediction():
    request_data = request.json
    features = get_features(request_data)
    return make_response(jsonify(predict(features)))


@app.route("/", methods=["GET"])
def index():
    return (
        "<p>Hello, This is a REST API used for Polyaxon ML Serving examples!</p>"
        "<p>Click the fullscreen button the get the URL of your serving API!<p/>"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
