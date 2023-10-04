"""Enables both the API and embedding loop to use shared `lodis` data strucutres."""
import lodis.hashmap
import lodis.queue
import lodis.small_priority_queue
import numpy as np

from services_common_code.config import USER_CONFIG


INPUT_DTYPE = np.dtype(
    [("job_id", np.int64), ("text_bytes", np.bytes_, USER_CONFIG.max_input_bytes)]
)
INPUT_QUEUE_SPEC = lodis.small_priority_queue.SmallPriorityQueueSpec.create(
    name="input_queue",
    item_dtype=INPUT_DTYPE,
    max_size=USER_CONFIG.max_input_queue_length,
)
RESULT_MAP_SPEC = lodis.hashmap.HashMapSpec.create(
    name="result_map",
    key_dtype=np.int64,
    value_dtype=np.dtype((np.float32, USER_CONFIG.embedding_dim)),
    max_size=3_000,
)
LOG_QUEUE_SPEC = lodis.queue.QueueSpec.create(
    name="log_queue", item_dtype=np.dtype("<U1000"), max_size=500
)
