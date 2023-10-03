import itertools
from typing import Iterable
from typing import List
from typing import TypeVar

T_element_type = TypeVar("T_element_type")


def iter_chunks(
    iterable: Iterable[T_element_type], max_chunk_size: int
) -> Iterable[List[T_element_type]]:
    """Iterate chunk-by-chunk through an iterable object."""
    iter_obj = iter(iterable)
    # Equivalent to:
    # while True:
    #     chunk = list(itertools.islice(iter_obj, max_chunk_size))
    #     if len(chunk) == 0:
    #         break
    #     yield chunk
    return iter(lambda: list(itertools.islice(iter_obj, max_chunk_size)), [])
