"""The data loaders collect the data from a variety of sources.

Each data loader is accessible by using its name as either a direct attribute
(`data_loaders.ParrotCounts`) or using its name with the `OPTIONS` dictionary
(`data_loaders.OPTIONS["ParrotCounts"]`). This allows for both hard-coded and
dynamic use of these variables.

Attributes:
    DataLoader: The abstract base class for all data loaders
    MockProfileLoader: A simple data loader which generates mocked data
    NPOStartProfileLoader: A data loader which loads from the NPO Start Govolte data

    OPTIONS: A dictionary which maps the names of each data loader to their
        respective class. It is automatically constructed to contain all of the
        imported subclasses of ``DataLoader``. So be sure to import any of the
        supported classes here to ease you development burdens.
"""
from __future__ import annotations

# Import several data loader classes
from .abstract_loader import DataLoader, LoaderType
from .mock_loader import MockProfileLoader
from ..utilities import safe_is_proper_subclass

# Collect all of the data loaders into a simple mapping
OPTIONS: dict[str, type[DataLoader]] = {
    name: obj for name, obj in globals().items() if safe_is_proper_subclass(obj, DataLoader)
}
