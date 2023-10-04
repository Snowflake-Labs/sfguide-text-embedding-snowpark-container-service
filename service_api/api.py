import asyncio
import base64
import logging
from typing import List
from typing import NamedTuple
from typing import NewType
from typing import Optional

import lodis.hashmap
import lodis.small_priority_queue
import numpy as np
from sfc_lru_cache import LRUCache
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

JobId = NewType("JobId", int)
B64EncodedFloat32Vector = NewType("B64EncodedFloat32Vector", str)


class JobIdCounter:
    def __init__(self):
        self._value = JobId(-1)
        self._lock = asyncio.Lock()

    async def get_next(self):
        async with self._lock:
            next_id = JobId(self._value + 1)
            self._value = next_id
        return next_id


class EmbedJob(NamedTuple):
    id: int
    utf8_encoded_text: bytes


SF_QUERY_ID_HEADER = "sf-external-function-current-query-id"
SF_QUERY_BATCH_ID_HEADER = "sf-external-function-query-batch-id"
# Even the first batch of a query is deprioritized if it is larger than this value.
LARGEST_PRIORITY_BATCH_SIZE = 1
QUERY_ID_CACHE_MAXSIZE = 10_000


#### Global state declaration.
_job_id_counter: Optional[JobIdCounter] = None
_input_queue: Optional[lodis.small_priority_queue.SmallPriorityQueue] = None
_result_map: Optional[lodis.hashmap.HashMap] = None
_query_id_cache: Optional[LRUCache] = None


#### Global state initialization.
def setup_state(
    embedding_input_queue_spec: lodis.small_priority_queue.SmallPriorityQueueSpec,
    embedding_result_map_spec: lodis.hashmap.HashMapSpec,
) -> None:
    global _job_id_counter, _input_queue, _result_map, _query_id_cache
    assert _job_id_counter is None
    assert _input_queue is None
    assert _result_map is None
    assert _query_id_cache is None

    # Initialize the job id counter.
    logger.info("Initializing job id counter")
    _job_id_counter = JobIdCounter()

    # Initialize query id cache.
    logger.info("Initializing query id cache")
    _query_id_cache = LRUCache(maxsize=QUERY_ID_CACHE_MAXSIZE)

    # Open the input queue and result hashmap.
    logger.info("Opening embedding input queue and result map")
    _input_queue = lodis.small_priority_queue.open(embedding_input_queue_spec)
    _result_map = lodis.hashmap.open(embedding_result_map_spec)


def teardown_state() -> None:
    global _input_queue, _result_map
    logger.info("Closing the API's access to the embedding input queue and result map")
    assert isinstance(_input_queue, lodis.small_priority_queue.SmallPriorityQueue)
    assert isinstance(_result_map, lodis.hashmap.HashMap)
    lodis.small_priority_queue.close(_input_queue)
    lodis.hashmap.close(_result_map)


#### Utility methods.
def _array_to_base64_float32(array: np.ndarray) -> B64EncodedFloat32Vector:
    return B64EncodedFloat32Vector(
        base64.b64encode(array.astype(np.float32).tobytes()).decode("utf8")
    )


#### API routes.
async def embed(request: Request) -> JSONResponse:
    # We know these are initialized, but the type checker doesn't.
    global _job_id_counter, _input_queue, _result_map, _query_id_cache
    assert isinstance(_job_id_counter, JobIdCounter)
    assert isinstance(_input_queue, lodis.small_priority_queue.SmallPriorityQueue)
    assert isinstance(_result_map, lodis.hashmap.HashMap)
    assert isinstance(_query_id_cache, LRUCache)

    logger.debug("Handling embedding")

    # Parse Snowflake external funtion call JSON structure.
    request_json = await request.json()
    query_id = request.headers.get(SF_QUERY_ID_HEADER, "UNKNOWN_QUERY")
    batch_id = request.headers.get(SF_QUERY_BATCH_ID_HEADER, "UNKNOWN_BATCH")
    row_numbers, texts = zip(*request_json["data"])

    # Check if this batch qualifies for priority embedding by being the first
    # batch of a query and by being small.
    is_seen_before_query_id = _query_id_cache.get(query_id, default=False)
    if not is_seen_before_query_id and len(texts) <= LARGEST_PRIORITY_BATCH_SIZE:
        priority = lodis.small_priority_queue.MOST_URGENT_PRIORITY
    else:
        priority = lodis.small_priority_queue.LEAST_URGENT_PRIORITY

    # Add/bump-to-latest the query id.
    _query_id_cache[query_id] = True

    # Queue up embedding.
    logger.info(
        f"[Query {query_id} | Batch {batch_id}] "
        f"Queueing embedding jobs at priority {priority}"
    )
    job_ids = []
    job_inputs: List[EmbedJob] = []
    for text in texts:
        job_id = await _job_id_counter.get_next()
        job_ids.append(job_id)
        job_inputs.append(EmbedJob(job_id, text.encode()))
    try:
        _input_queue.put_batch_nowait(job_inputs, priority=priority)
    except lodis.small_priority_queue.NotEnoughSpace:
        # If the queue is full, the service is overwhelmed, so let's return 429.
        # NOTE: This can unfortunately break the "fast lane" priority treatment.
        # TODO: Consider moving to two separate queues to reserve more space for
        # priority queries.
        return JSONResponse(content="", status_code=429)

    # Wait for results.
    results = []
    for job_id in job_ids:
        embedding_vector = await _result_map.pop_block_async(job_id)
        embedding_b64_encoded = _array_to_base64_float32(embedding_vector)
        results.append(embedding_b64_encoded)

    assert len(row_numbers) == len(results)

    # Format a JSON response satisfying external function response structure.
    response_json = {"data": list(zip(row_numbers, results))}

    return JSONResponse(response_json)


async def healthcheck(request: Request) -> JSONResponse:
    return JSONResponse({"success": True})


ROUTES = [
    Route("/embed", embed, methods=["POST"]),
    Route("/healthcheck", healthcheck, methods=["GET"]),
]
