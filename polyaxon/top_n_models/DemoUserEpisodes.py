from __future__ import annotations

import datetime
import random

from .abstract_top_n_model import ModelTypeEnum, TopNModel, TopNOutputKeys


class DemoUserEpisodes(TopNModel):
    """Demonstration implementation of a EPISODES mode.

    Example:
    >>> from ..data_loaders import MockProfileLoader
    >>> m = DemoUserEpisodes(data_loader = MockProfileLoader(max_datetime=datetime.datetime(2021,11,1,9,0)))
    >>> m = m.fit()
    >>> now = datetime.datetime.now()
    >>> recs = m.predict(now, from_ids=['a','b','c'], n=10)
    >>> len(recs) == 3
    True
    >>> len(set(recs[0].keys()).intersection({'from_key','items','scores','from_key_type','to_key_type','recommender'})) == 6 # noqa
    True
    >>> type(recs[0]['scores'][0]) == float
    True
    >>> now.isoformat()==recs[0]['datetime_context']
    True
    >>> created = datetime.datetime.fromisoformat(recs[0]['datetime_created'])
    >>> type(created)
    <class 'datetime.datetime'>


    """

    model_type = ModelTypeEnum.USER_TO_ITEM
    output_to_field = TopNOutputKeys.EPISODE_ID

    def fit(self) -> TopNModel:
        """For the demo, fit only entails setting a mock space of 'known_items'."""
        self.known_items = {f"i-{i}" for i in range(100_000)}
        return self

    def predict(
        self, timestamp: datetime.datetime, from_ids: list[str], n: int
    ) -> list[dict[str, str | list[float] | list[str]]]:
        """Generate predictions based on the `fit` self.known items, the white and blacklist.

        Predictions are based on random betavariate scores.

        See abstract class for interface information.
        """
        recommendations: list[dict[str, str | list[float] | list[str]]] = []

        for a in from_ids:
            creation_time = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
            pred_items = random.choices(list(self.known_items), k=min(n, len(self.known_items)))
            pred_scores = [float(random.betavariate(1, 1)) for _ in range(len(pred_items))]
            recommendations.append(
                {
                    "from_key": a,
                    "from_key_type": self.output_from_field.value,
                    "to_key_type": self.output_to_field.value,
                    "scores": pred_scores,
                    "items": pred_items,
                    "datetime_context": timestamp.isoformat(),
                    "datetime_created": creation_time,
                    "recommender": self.name,
                }
            )

        return recommendations
