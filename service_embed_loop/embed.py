"""This is a stub that should be replaced by user code."""
import logging
from typing import Callable
from typing import Sequence

import numpy as np


def get_embed_fn(logger: logging.Logger) -> Callable[[Sequence[str]], np.ndarray]:
    raise NotImplementedError(
        "Looks like the build got messed up no `embed.py` was pulled in."
    )
