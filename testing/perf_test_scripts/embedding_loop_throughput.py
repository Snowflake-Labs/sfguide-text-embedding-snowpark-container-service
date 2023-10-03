import logging
import sys
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Iterator
from typing import Tuple

import embed_lib.e5
import lodis.hashmap
import lodis.librt_semaphore
import lodis.queue
import lodis.small_priority_queue
import numpy as np
import pandas as pd

# Enable imports of `src` by patching the PYTHONPATH.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embedding_worker import EmbeddingProcessParameters
from src.embedding_worker import shutdown_embedding_process
from src.embedding_worker import startup_embedding_process
from services.e5_service.src.shared.multiprocess_logging import start_listening_for_logs
from services.e5_service.src.shared.multiprocess_logging import stop_listening_for_logs

logger = logging.getLogger(__name__)

N_PER_BATCH = 140
N_BATCHES = 3
MAX_BATCH_SIZE = 4
MODEL_NAME = "base"


def benchmark_batch(texts, input_queue, result_map):
    put_times = []
    wait_times = []
    pop_times = []
    for i, item in enumerate(texts):
        # Time put.
        t_start = perf_counter()
        input_queue.put((i, item))
        t_end = perf_counter()
        put_times.append(t_end - t_start)
    for i in range(len(texts)):
        # Time wait.
        t_start = perf_counter()
        while i not in result_map:
            sleep(0.005)
        t_end = perf_counter()
        wait_times.append(t_end - t_start)

        # Time pop.
        t_start = perf_counter()
        _ = result_map.pop(i)
        t_end = perf_counter()
        pop_times.append(t_end - t_start)
        # logger.info(f"Got embedding {res}")
    return put_times, wait_times, pop_times


def main() -> None:
    texts = ["passage: " + "hello world, this is a sentence!" * 10] * N_PER_BATCH
    result_times = []
    with setup_context(max_batch_size=MAX_BATCH_SIZE, e5_model_name=MODEL_NAME) as (
        input_queue,
        result_map,
    ):
        # Wait for the process to come up.
        sleep(5)
        for _ in range(N_BATCHES):
            result_times.append(benchmark_batch(texts, input_queue, result_map))
    for i, name in enumerate(("put", "wait", "pop")):
        df = pd.DataFrame({j: res[i] for j, res in enumerate(result_times)})
        print(name)
        print("median")
        print(df.median())
        print("p95")
        print(df.quantile(0.95))
        print("grand mean")
        print(np.mean(df.to_numpy().ravel()))
        print("grand median")
        print(np.median(df.to_numpy().ravel()))
        print("grand p95")
        print(np.quantile(df.to_numpy().ravel(), 0.95))
        print()
        print()

    # Try direct embedding.
    model = embed_lib.e5.load_e5_model(MODEL_NAME)
    t_start = perf_counter()
    embed_lib.e5.embed(model, texts, batch_size=MAX_BATCH_SIZE)
    t_end = perf_counter()
    duration = t_end - t_start
    duration_per_text = duration / len(texts)
    print(f"Straight embed took: {duration:.4f}, or {duration_per_text:.4f} per text")


@contextmanager
def setup_context(
    max_batch_size: int = 8, e5_model_name: str = "base"
) -> Iterator[
    Tuple[lodis.small_priority_queue.SmallPriorityQueue, lodis.hashmap.HashMap]
]:
    INPUT_TRUNC_LEN = 10240
    EMBEDDING_DIM = 768
    INPUT_DTYPE = np.dtype(
        [("job_id", np.int64), ("text_bytes", np.bytes_, INPUT_TRUNC_LEN)]
    )
    INPUT_QUEUE_SPEC = lodis.small_priority_queue.SmallPriorityQueueSpec.create(
        name="input_queue", item_dtype=INPUT_DTYPE, max_size=300
    )
    RESULT_MAP_SPEC = lodis.hashmap.HashMapSpec.create(
        name="result_map",
        key_dtype=np.int64,
        value_dtype=np.dtype((np.float32, EMBEDDING_DIM)),
        max_size=int(N_PER_BATCH * 1.5),
    )
    LOG_QUEUE_SPEC = lodis.queue.QueueSpec.create(
        name="log_queue", item_dtype=np.dtype("<U1000"), max_size=50
    )
    EMBEDDING_PROCESS_PARAMETERS = EmbeddingProcessParameters(
        log_queue_spec=LOG_QUEUE_SPEC,
        input_queue_spec=INPUT_QUEUE_SPEC,
        result_map_spec=RESULT_MAP_SPEC,
        e5_model_name=e5_model_name,
        max_batch_size=max_batch_size,
    )

    # Clean up possibly messed up semaphores.
    _ensure_q_not_allocated(INPUT_QUEUE_SPEC.name)
    _ensure_q_not_allocated(LOG_QUEUE_SPEC.name)
    _ensure_hm_not_allocated(RESULT_MAP_SPEC.name)

    logger.info("Allocating lodis stuff")
    lodis.small_priority_queue.allocate(INPUT_QUEUE_SPEC)
    lodis.hashmap.allocate(RESULT_MAP_SPEC)
    lodis.queue.allocate(LOG_QUEUE_SPEC)
    logger.info("Starting log listener")
    start_listening_for_logs(LOG_QUEUE_SPEC)
    sleep(1)
    logger.info("Starting embedding process")
    startup_embedding_process(EMBEDDING_PROCESS_PARAMETERS)
    sleep(3)
    input_queue = lodis.small_priority_queue.open(INPUT_QUEUE_SPEC)
    result_map = lodis.hashmap.open(RESULT_MAP_SPEC)
    try:
        yield input_queue, result_map
    finally:
        lodis.small_priority_queue.close(input_queue)
        lodis.hashmap.close(result_map)
        shutdown_embedding_process()
        stop_listening_for_logs()
        lodis.queue.unlink(LOG_QUEUE_SPEC)
        lodis.small_priority_queue.unlink(INPUT_QUEUE_SPEC)
        lodis.hashmap.unlink(RESULT_MAP_SPEC)


def _ensure_hm_not_allocated(name: str):
    fake_id = lodis.hashmap.HashMapSpec(name, np.dtype(np.bool_), 1)
    try:
        memory = SharedMemory(fake_id.memory_name)
        memory.close()
        memory.unlink()
        del memory
    except Exception:
        pass
    try:
        lodis.librt_semaphore.unlink(fake_id.lock_name)
    except Exception:
        pass


def _ensure_q_not_allocated(name: str):
    fake_id = lodis.queue.QueueSpec(name, np.dtype(np.bool_), 1)
    try:
        memory = SharedMemory(fake_id.memory_name)
        memory.close()
        memory.unlink()
        del memory
    except Exception:
        pass
    try:
        lodis.librt_semaphore.unlink(fake_id.lock_name)
        lodis.librt_semaphore.unlink(fake_id.notempty_sem_name)
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s [%(levelname)s]: %(message)s"
    )
    main()
