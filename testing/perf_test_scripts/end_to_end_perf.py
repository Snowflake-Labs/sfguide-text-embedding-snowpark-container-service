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

ENDPOINT = "http://localhost:8000/embed"
EXAMPLE_TEXT = Path(__file__).resolve().parent.joinpath("example_text.txt").read_text()
BATCH_SIZE = 10
N_BATCHES = 30
N_WORDS_IN_BATCHED_TEXTS = 600
N_WORDS_IN_SHORT_TEXTS = 5
N_BULK_QUERIES = 3
SLEEP_BETWEEN_INTERMITTENT_QUERIES = 20


async def main() -> None:
    intermittent_task = asyncio.create_task(
        intermittent_quick_queries(sleep_sec=SLEEP_BETWEEN_INTERMITTENT_QUERIES)
    )
    bulk_query_tasks = [
        asyncio.create_task(bulk_query(N_BATCHES)) for _ in range(N_BULK_QUERIES)
    ]
    bulk_times_list = [pd.Series(await task) for task in bulk_query_tasks]
    intermittent_task.cancel()
    intermittent_times = pd.Series(await intermittent_task)
    print_results(bulk_times_list, intermittent_times)


def print_results(
    bulk_times_list: Sequence[pd.Series], intermittent_times: pd.Series
) -> None:
    total_words = N_BULK_QUERIES * BATCH_SIZE * N_BATCHES * N_WORDS_IN_BATCHED_TEXTS
    print(
        f"Bulk embedded {total_words:,d} words as {N_BULK_QUERIES} simultaneous "
        f"bulk queries running {N_BATCHES} batches of {BATCH_SIZE} texts "
        f"containing exactly {N_WORDS_IN_BATCHED_TEXTS} words each (truncation "
        "may have been applied)."
    )
    print(
        f"At the same time, periodically executed {len(intermittent_times)} "
        f"single-text {N_WORDS_IN_SHORT_TEXTS}-word queries."
    )
    bulk_total_times = pd.Series([s.sum() for s in bulk_times_list])
    bulk_batch_times = pd.concat(bulk_times_list)
    print(
        f"Bulk query total execution times (in sec): {bulk_total_times.round(1).tolist()}"
    )
    print(
        "Bulk queries' per-batch execution time stats (in sec): μ ± 2σ = "
        f"{bulk_batch_times.mean():.2f} ± {2 * bulk_batch_times.std(): .2f} | "
        f"max = {bulk_batch_times.max():.2f}"
    )
    print(
        "Intermittent query execution time stats (in sec): μ ± 2σ = "
        f"{intermittent_times.mean():.2f} ± {2 * intermittent_times.std(): .2f} | "
        f"max = {intermittent_times.max():.2f}"
    )


async def bulk_query(
    n_batches: int,
    n_texts_per_batch: int = BATCH_SIZE,
    n_words_per_text: int = N_WORDS_IN_BATCHED_TEXTS,
) -> Sequence[float]:
    async with aiohttp.ClientSession() as session:
        batch_times = [
            await time_embedding(
                session=session, headers=headers, json_payload=json_payload
            )
            for headers, json_payload in iterate_mock_query_requests(
                n_batches=n_batches,
                n_texts_per_batch=n_texts_per_batch,
                n_words_per_text=n_words_per_text,
            )
        ]
    return batch_times


async def intermittent_quick_queries(
    n_words: int = N_WORDS_IN_SHORT_TEXTS, sleep_sec: float = 0.7
) -> Sequence[float]:
    result_times = []
    try:
        while True:
            req_iter = iterate_mock_query_requests(
                n_batches=1, n_texts_per_batch=1, n_words_per_text=n_words
            )
            headers, json_payload = next(iter(req_iter))
            async with aiohttp.ClientSession() as session:
                embed_time = await time_embedding(
                    session=session, headers=headers, json_payload=json_payload
                )
            result_times.append(embed_time)
            await asyncio.sleep(sleep_sec)
    finally:
        return result_times


async def time_embedding(
    session: aiohttp.ClientSession,
    headers: Dict[str, str],
    json_payload: Dict[Any, Any],
) -> float:
    start = time.perf_counter()
    is_done = False
    duration = float("nan")
    while not is_done:
        async with session.request(
            "POST", ENDPOINT, json=json_payload, headers=headers
        ) as result:
            if result.status == 429:
                await asyncio.sleep(0.2)
                continue
            else:
                is_done = True
            await result.json()
        end = time.perf_counter()
        duration = end - start
    return duration


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
