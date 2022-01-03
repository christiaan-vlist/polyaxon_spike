"""A CLI for the recommendation engine of Data and Personalisatie.

Be sure to read `README.md` or `main.py` for a more in-depth discussion. This
file ensures all sub-packages are available for any importers.

Attributes:
    commands: The code for running the more complex commands in this CLI
    data_loaders: The data loaders collect the data from a variety of sources
    pipelines: The pipeline factories which create our scikit-learn pipelines
    sideload_transformers: Transformers which load in additional data
    transformers: A collection of data transformers
    utilities: Some isolated functions and classes which simplify our code base
"""
# flake8: noqa F401
from __future__ import annotations

from . import data_loaders
from . import top_n_models
from . import utilities
from . import train
