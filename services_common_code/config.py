"""Enables both the API and embedding loop to use shared `lodis` data strucutres."""
from typing import NamedTuple

import lodis.hashmap
import lodis.queue
import lodis.small_priority_queue
import numpy as np


class EmbeddingProcessParameters(NamedTuple):
    log_queue_spec: lodis.queue.QueueSpec
    input_queue_spec: lodis.small_priority_queue.SmallPriorityQueueSpec
    result_map_spec: lodis.hashmap.HashMapSpec
    e5_model_name: str
    max_batch_size: int


EMBED_MAX_BATCH_SIZE = 4
E5_MODEL_NAME = "base"
# Enough for 512 18-character tokens with 2 whitespace characters each.
INPUT_TRUNC_LEN = 10240
EMBEDDING_DIM = 768
INPUT_DTYPE = np.dtype(
    [("job_id", np.int64), ("text_bytes", np.bytes_, INPUT_TRUNC_LEN)]
)
# NOTE: Docker defaults to setting `/dev/shm` to only 64MiB, so
# the total size of the input queue and result map need to be pretty small.
INPUT_QUEUE_SPEC = lodis.small_priority_queue.SmallPriorityQueueSpec.create(
    name="input_queue", item_dtype=INPUT_DTYPE, max_size=16
)
RESULT_MAP_SPEC = lodis.hashmap.HashMapSpec.create(
    name="result_map",
    key_dtype=np.int64,
    value_dtype=np.dtype((np.float32, EMBEDDING_DIM)),
    max_size=3_000,
)
LOG_QUEUE_SPEC = lodis.queue.QueueSpec.create(
    name="log_queue", item_dtype=np.dtype("<U1000"), max_size=500
)
EMBEDDING_PROCESS_PARAMETERS = EmbeddingProcessParameters(
    log_queue_spec=LOG_QUEUE_SPEC,
    input_queue_spec=INPUT_QUEUE_SPEC,
    result_map_spec=RESULT_MAP_SPEC,
    e5_model_name=E5_MODEL_NAME,
    max_batch_size=EMBED_MAX_BATCH_SIZE,
)
