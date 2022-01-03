from __future__ import annotations

import collections.abc
import datetime
import random
import typing

import joblib
import pandas
import scipy.stats
import tqdm

from abstract_loader import DataLoader, LoaderType


class MockProfileLoader(DataLoader):
    """A mocked data loader for testing purposes."""

    loader_type = LoaderType.PROFILE_ID

    genremap: dict[str, list[str]] = {}

    def _load_interactions(
        self,
        from_dt: datetime.datetime,
        until_dt: datetime.datetime,
        n: typing.Optional[int] = 1000,
    ) -> pandas.DataFrame:
        """See base class."""
        # Convert the given time range
        start_time = from_dt.timestamp()
        end_time = until_dt.timestamp()

        hours_covered = (end_time - start_time) / 60 / 60

        # Set some random parameters
        n_items = 300 * round(hours_covered)
        n_users = 800 * round(hours_covered) if n is None else n
        n_choices = 5
        n_types = 4

        # Generate some random preferences
        prefs = {i: scipy.stats.nbinom.rvs(n=0.7, p=0.5, size=n_items) for i in range(n_types)}

        # Construct the new Pandas dataframe
        items = []

        def generate_interactions(
            user_id: int, start_time: float, end_time: float, n_types: int, n_choices: int
        ) -> list[dict[str, typing.Any]]:
            item_ids = random.choices(range(n_items), prefs[user_id % n_types], k=n_choices)
            return [
                {
                    "timestamp": datetime.datetime.fromtimestamp(random.randrange(int(start_time), int(end_time))),
                    "user": f"user_{user_id}",
                    "item": f"item_{item_id}",
                    "weight": prefs[user_id % n_types][item_id] / max(prefs[user_id % n_types]),
                }
                for item_id in item_ids
            ]

        items = joblib.Parallel(-1)(
            joblib.delayed(generate_interactions)(user_id, start_time, end_time, n_types, n_choices)
            for user_id in tqdm.tqdm(range(n_users), desc="generating mock data", total=n_users)
        )

        # Return the results back to the caller
        return pandas.DataFrame([item for sublist in items for item in sublist])

    def load_genres(
        self,
        content_ids: collections.abc.Sequence[str],
    ) -> collections.abc.Iterator[list[str]]:
        """See base class."""
        for prid in content_ids:
            if prid in self.genremap:
                yield self.genremap[prid]
                continue
            genres = random.choices(
                ["drama", "actualiteiten", "documentaire", "spanning"],
                k=random.randint(1, 2),
            )
            self.genremap[prid] = genres
            yield genres
