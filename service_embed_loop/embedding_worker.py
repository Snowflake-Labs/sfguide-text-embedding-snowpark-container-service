import logging
import signal
from queue import Empty
from typing import Callable
from typing import List
from typing import Sequence

import lodis.hashmap
import lodis.queue
import lodis.small_priority_queue
import numpy as np
from embed import get_embed_fn
from multiprocess_logging import LodisQueueLogHandler
from multiprocess_logging import setup_root_handler

from services_common_code import lodis_configs
from services_common_code.config import USER_CONFIG


def main() -> None:
    """This function constitutes a parameterized embedding program to be run in
    a separate Python process.
    """

    # Set up logging.
    logger = logging.getLogger(__name__)
    log_handler = LodisQueueLogHandler(lodis_configs.LOG_QUEUE_SPEC)
    setup_root_handler(log_handler, level=logging.INFO)
    logger.info("Embed loop starting")

    # Open access to the `lodis`-based shared input queue and result hashmap.
    logger.info("Opening input queue and result map")
    input_queue = lodis.small_priority_queue.open(lodis_configs.INPUT_QUEUE_SPEC)
    result_map = lodis.hashmap.open(lodis_configs.RESULT_MAP_SPEC)

    # Set up signal handling so we can terminate the loop gracefully.
    is_looping = [True]  # We box this bool in a list so we can pass it by reference.

    def _handle_signal(*_):
        nonlocal is_looping, input_queue
        is_looping[0] = False

        # Add a "flush" item to the queue in case the loop is stuck waiting for
        # the next item.
        input_queue.put((-1, "done"))

    logger.info("Setting up embedding process SIGTERM handling")
    signal.signal(signal.SIGTERM, _handle_signal)

    # Run the embedding loop.
    embed_fn = get_embed_fn(logger=logger)
    _embedding_loop(
        logger=logger,
        embed_fn=embed_fn,
        is_looping=is_looping,
        input_queue=input_queue,
        result_map=result_map,
        max_batch_size=USER_CONFIG.max_batch_size,
    )

    # Cleanup when signaled to terminate the loop.
    logger.info("Embedding loop terminated, cleaning up resources")
    lodis.small_priority_queue.close(input_queue)
    lodis.hashmap.close(result_map)
    logger.info("Input queue and result map closed, now closing log handler")
    logging.root.removeHandler(log_handler)
    log_handler.close()


def _embedding_loop(
    logger: logging.Logger,
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    is_looping: List[bool],
    input_queue: lodis.small_priority_queue.SmallPriorityQueue,
    result_map: lodis.hashmap.HashMap,
    max_batch_size: int,
) -> None:
    assert len(is_looping) == 1
    logging.info("Starting embedding loop")
    while is_looping[0]:
        logger.debug("Awaiting the next batch of texts to embed.")

        # Get an input batch of up to EMBED_MAX_BATCH_SIZE from the queue.
        batch = [input_queue.get(block=True)]
        for _ in range(max_batch_size - 1):
            try:
                next_item = input_queue.get(block=False)
                batch.append(next_item)
            except Empty:
                break
        logger.debug(f"Got batch of {len(batch)} items")

        # Run embeddings.
        texts = tuple(item["text_bytes"].decode() for item in batch)
        result_array = embed_fn(texts)
        assert result_array.shape[0] == len(batch)

        # Store results.
        logger.debug("Embedding finished, storing result in result map")
        for i, item in enumerate(batch):
            job_id = item["job_id"]
            # We use the blocking version of `__setitem__` to gracefully wait
            # for space if the result map gets full.
            result_map.set_block(job_id, result_array[i, :])


if __name__ == "__main__":
    main()
