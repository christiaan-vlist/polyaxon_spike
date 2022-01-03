from __future__ import annotations

import abc
import collections.abc
import datetime
import enum
import logging
import typing

import pandas

from const import get_dataframe_dtypes


class LoaderType(str, enum.Enum):
    """Enum to determine what the model is trained to take as 'from' key."""

    PROFILE_ID = "profile_id"


class DataLoader(abc.ABC):
    """A semi-optimized abstract base class for all data loaders.

    This class also contains several type and sanity checks on some of the data
    returned by subclasses. This ensures that certain kinds of data type issues
    are caught before they turn into real-world issues.

    Methods:
        load_interactions: Load (user, item)-interactions
        _load_interactions: Load (user, item)-interactions (back-end)
        load_genres: Iterate over the genres of the given content IDs

    Attributes:
        max_datetime : A maximum datetime that limits data access for temporal
                       train/test splits. Defaults to None (e.g. current time
                       for each call).
    """

    loader_type: LoaderType

    @typing.final
    def __init__(self, max_datetime: datetime.datetime | None = None):
        """Create a dataloader.

        Arguments:
            max_datetime : a limit set to the latest data accessible to callers,
                           used to enforce temporal test-train splits. Will allow
                           access to latest data (per call) when None.
        """
        self.max_datetime = max_datetime
        logging.debug(f"Initializing dataloader up to {self.max_datetime}")

    @typing.final
    def load_interactions(
        self,
        from_dt: datetime.datetime | None = None,
        until_dt: datetime.datetime | None = None,
        from_offset: datetime.timedelta | None = None,
        num_records: typing.Optional[int] = 1000,
    ) -> pandas.DataFrame:
        """Load the (user, item)-interactions from the underlying data source.

        Note that this function might return a rather large dataset. Developers
        are therefore advised to limit the resulting size and to avoid copying
        the result wherever possible.

        Args:
            from_offset: How much older interactions can be relative to the maximum
                         available timestamp set during initialization.
            from_dt: The start point in time to load interactions from. overrides
                     `from_offset`.
            until_dt: The last date/time to collect the information of
            num_records: The maximum number of interactions to return, defaults to 1000

        Returns:
            The interactions collected in a pandas dataframe
        """
        if self.max_datetime and until_dt:
            until_dt = min(self.max_datetime, until_dt)
        else:
            until_dt = self.max_datetime or datetime.datetime.now()

        if from_dt is None:
            assert from_offset is not None, ValueError("Either `from_dt` or `from_offset` should be specified.")
            from_dt = until_dt - from_offset

        i = self._load_interactions(from_dt, until_dt, num_records)
        return i.astype(
            get_dataframe_dtypes(),
            copy=False,
        )

    @abc.abstractmethod
    def _load_interactions(
        self,
        from_dt: datetime.datetime,
        until_dt: datetime.datetime,
        num_records: typing.Optional[int] = 1000,
    ) -> pandas.DataFrame:
        """Load the (user, item)-interactions from the underlying data source.

        Note that this function might return a rather large dataset. Developers
        are therefore advised to limit the resulting size and to avoid copying
        the result wherever possible.

        Args:
            from_dt The first date/time to collect the information of
            until_dt: The last date/time to collect the information of
            num_records: The maxium number of interactions to return

        Returns:
            The interactions collected in a pandas dataframe
        """

    @abc.abstractmethod
    def load_genres(
        self,
        content_ids: collections.abc.Sequence[str],
    ) -> collections.abc.Iterator[collections.abc.Sequence[str]]:
        """Iterate over the genres associated with the given content IDs.

        Args:
            content_ids: The content IDs for which genre labels are requested

        Yields:
            A sequence of genre strings for a content ID in order
        """
