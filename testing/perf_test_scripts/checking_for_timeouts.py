# Requirements: `pandas` and `aiohttp`.
import asyncio
import logging
import re
import time
from pathlib import Path
from random import Random
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Sequence
from typing import Tuple
from uuid import uuid4

import aiohttp
import pandas as pd
from tqdm import tqdm

ENDPOINT = "http://localhost:8000/embed"
EXAMPLE_TEXT = Path(__file__).resolve().parent.joinpath("example_text.txt").read_text()
BATCH_SIZE = 10
N_BATCHES_PER_QUERY = 3
N_WORDS_PER_ROW = 60
N_SIMULTANEOUS_QUERIES = 20
RETRY_WAIT_SEC = 0.2


async def main() -> None:
    with tqdm(total=BATCH_SIZE * N_BATCHES_PER_QUERY * N_SIMULTANEOUS_QUERIES) as pbar:
        query_tasks = [
            asyncio.create_task(run_query(N_BATCHES_PER_QUERY, pbar))
            for _ in range(N_SIMULTANEOUS_QUERIES)
        ]
        results = [await task for task in query_tasks]
        df_res_map = {}
        for i, result in enumerate(results):
            times, last_attempt_times, retries = zip(*result)
            df_res_map[i] = pd.DataFrame(
                {
                    "query_time": times,
                    "last_request_time": last_attempt_times,
                    "retries": retries,
                }
            )
        df_res = pd.concat(df_res_map, axis=0, names=["query"])
        df_res.to_csv("timeout_check_results.csv")


async def run_query(
    n_batches: int,
    pbar: tqdm,
    n_texts_per_batch: int = BATCH_SIZE,
    n_words_per_text: int = N_WORDS_PER_ROW,
) -> Sequence[Tuple[float, float, int]]:
    batch_results = []
    async with aiohttp.ClientSession() as session:
        for headers, json_payload in iterate_mock_query_requests(
            n_batches=n_batches,
            n_texts_per_batch=n_texts_per_batch,
            n_words_per_text=n_words_per_text,
        ):
            batch_results.append(
                await time_embedding(
                    session=session, headers=headers, json_payload=json_payload
                )
            )
            pbar.update(BATCH_SIZE)
    return batch_results


async def time_embedding(
    session: aiohttp.ClientSession,
    headers: Dict[str, str],
    json_payload: Dict[Any, Any],
) -> Tuple[float, float, int]:
    start = time.perf_counter()
    retries = 0
    duration = float("nan")
    last_attempt_duration = float("nan")
    for _attempt in range(99999):
        attempt_start = time.perf_counter()
        async with session.request(
            "POST", ENDPOINT, json=json_payload, headers=headers
        ) as result:
            assert result.status is not None
            if result.status == 429:
                retries += 1
                await asyncio.sleep(RETRY_WAIT_SEC)
                continue
            await result.json()
        end = time.perf_counter()
        duration = end - start
        last_attempt_duration = end - attempt_start
        break
    return duration, last_attempt_duration, retries


def iterate_mock_query_requests(
    n_batches: int, n_texts_per_batch: int, n_words_per_text: int
) -> Iterable[Tuple[Dict[str, str], Dict[Any, Any]]]:
    text_iter = iter(_infinite_texts(n_words=n_words_per_text))
    mock_query_id = str(uuid4())
    for _ in range(n_batches):
        mock_batch_id = str(uuid4())
        json_payload = {
            "data": [[i, next(text_iter)] for i in range(n_texts_per_batch)]
        }
        headers = {
            "sf-external-function-current-query-id": mock_query_id,
            "sf-external-function-query-batch-id": mock_batch_id,
        }
        yield headers, json_payload


def _infinite_texts(n_words: int, seed: int = 0) -> Iterable[str]:
    word_matches = list(re.finditer(r"\w+", EXAMPLE_TEXT))
    wiggle_room = len(word_matches) - n_words
    assert wiggle_room > 0
    rng = Random(seed)
    while True:
        start_word = rng.randint(0, wiggle_room - 1)
        end_word = start_word + n_words
        start_char_idx = word_matches[start_word].start()
        end_char_idx = word_matches[end_word].end()
        yield EXAMPLE_TEXT[start_char_idx:end_char_idx]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s [%(levelname)s]: %(message)s"
    )
    asyncio.run(main())
