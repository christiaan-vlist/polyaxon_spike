"""The abstract class for top-n models."""

from __future__ import annotations

import abc
import dataclasses
import datetime
import enum
import typing

from .. import data_loaders
from .. import utilities


class TopNOutputKeys(str, enum.Enum):
    """Output fieldnames for top-n-models."""

    EPISODE_ID = "episode_id"
    SERIES_ID = "series_id"


class ModelTypeEnum(str, enum.Enum):
    """An enumeration over all of the possible model types. This can be used for evaluation purposes.

    Attributes:
        USER_TO_ITEM: A pipeline which operates on (user, item)-interactions
        ITEM_TO_ITEM: A pipeline which operates on (item, item)-interactions
    """

    USER_TO_ITEM = "user_to_item"
    ITEM_TO_ITEM = "item_to_item"


@dataclasses.dataclass
class TrainedTopNModel:
    """Class for persisting top-n models.

    Attributes:
        name: The standardized name of the trained model
        model: The trained model belonging to this model
        model_type: The type of data used in the model
        input_type: Input key such as profile id
        output_type: Output key such as series id
        timestamp: The time at which this model has been trained
    """

    name: str
    model: TopNModel
    model_type: ModelTypeEnum
    input_type: data_loaders.LoaderType
    output_type: TopNOutputKeys
    timestamp: datetime.datetime


class TopNModel(abc.ABC):
    """Top-N model abstract class.

    Top-N models are trained to return top-n from-to relevance objects.
    What constitutes the 'from' key depends on the dataloader, which
    means the same TopNModel can be applied to NPOStart Profiles,
    NPOStart party_ids, NPOLuister Series etcetera, depending on the
    dataloader used to instantiate it.

    Attributes:
        model_type        : Type of interaction of the model.
        output_to_field   : set in concrete implementations, provides the
                            key to which relevance scores pertain, such
                            as the 'series_id' or 'episode_id'.
        output_from_field : the input-key for predictions, based on the
                            dataloader provided on initialization. Used
                            in the output as the key for the output ids
                            (e.g. {..., self.output_from_field: "MID_123"})

    Notes:
        - access to data should always move through the provided dataloader
        - it's the responsibility of the `fit` method to maintain the
          boundaries of the `from_datetime` and `to_datetime`.

    """

    model_type: ModelTypeEnum

    output_to_field: TopNOutputKeys

    # set during initialization
    output_from_field: data_loaders.LoaderType
    name: str

    @typing.final
    def __init__(self, data_loader: data_loaders.DataLoader, hyperparameters: dict[str, typing.Any] | None = None):
        """Initialize a top-n model with the dataloader.

        Params:
            data_loader : The sole source of data for this model.
                          Seriously, don't use another source.
                          Not even indirectly.
            **kwargs    : hyperparameters for the model, including
                          the amout of training data to use.
        """
        self.data_loader = data_loader
        # set the outputfield based on the dataloader
        # dataloaders should thus determine the basis for
        # training (e.g. NPOStart profiles or NPOLuister party_ids)
        self.output_from_field = data_loader.loader_type

        self.hyperparameters = hyperparameters or {}

        self.name = "test"

    @abc.abstractmethod
    def fit(self) -> TopNModel:
        """Fit to observations from the data_loader using data from the provided timeframe."""

    @abc.abstractmethod
    def predict(
        self, timestamp: datetime.datetime, from_ids: list[str], n: int
    ) -> list[dict[str, str | list[float] | list[str]]]:
        """Generate top-n lists for each anchor (e.g. user or item).

        Args:
            timestamp : The target time for the recommendations to be served
            anchors   : A list of string ids, each of which is the bases for a top-n list, e.g.
                        the users or items for which the lists are generated.
            n         : The maximum number of recommendations to return for each user

        Returns:
            list of dicts containing the predictions in the format:
              [
                  {
                    "from_key"           : from_id,
                    "from_key_type"      : self.output_from_key.value
                    "to_key_type"        : self.output_to_key.value
                    "scores"             : [float(score), ...]
                    "items"              : ["item_1", ....]
                    "recommender"        : self.name,
                    "datetime_context"   : timestamp.isoformat(),
                    "datetime_created"   : (time of prediction).isoformat(),

                },
              ...
              ]

        Note:
            - Predict-time performance (speed) is essential, so bear this in mind. This is also the
              reason why the return object should be created as part of the model implementation.
            - Do not forget to provide the timezone in the timestamps!
        """

    @typing.final
    def to_trained(self) -> TrainedTopNModel:
        """Return the trained representation of the model."""
        return TrainedTopNModel(
            name=self.name,
            model=self,
            model_type=self.model_type,
            input_type=self.output_from_field,
            output_type=self.output_to_field,
            timestamp=datetime.datetime.now(),
        )
