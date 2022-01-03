"""Top N Models.

Contains the abstract class and implementations thereof.

Top N Models provide recommendations based on the Dataloader `from` key
(e.g. NPOStartProfileLoader wil have 'profile_id' based recommendations)
and specifies it's own output key (e.g. 'episodes' or 'series'). Output
is oriented towards getting top-n item-relevance pairs for a set of
input keys (e.g. profile_ids, series_ids etcetera, depending on the dataloader).

See the abstract class for details.

"""

from ..utilities import safe_is_proper_subclass
from .abstract_top_n_model import TopNModel, TrainedTopNModel, ModelTypeEnum  # noqa

from .DemoUserEpisodes import DemoUserEpisodes  # noqa

OPTIONS = {name: obj for name, obj in globals().items() if safe_is_proper_subclass(obj, TopNModel)}
