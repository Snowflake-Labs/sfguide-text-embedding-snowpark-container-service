import asyncio
import contextlib
import logging
import warnings
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from queue import Full
from time import sleep
from typing import Generic
from typing import Iterator
from typing import Sequence

import numba
import numpy as np
from lodis import librt_semaphore
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from ._constants import POLL_SLEEP_DURATION
from ._typing import NonscalarNDArray
from ._typing import T_Item
from ._typing import T_ItemDtype

logger = logging.getLogger(__name__)

# More type declarations.
ItemArray = NonscalarNDArray[T_ItemDtype]
PriorityArray = NDArray[np.uint8]
AgeArray = NDArray[np.uint64]
MaskArray = NDArray[np.bool_]
AGE_DTYPE = np.dtype(np.uint64)
PRIORITY_DTYPE = np.dtype(np.uint8)
MOST_URGENT_PRIORITY = 1
LEAST_URGENT_PRIORITY = 255


class NotEnoughItems(Exception):
    """Raised if attempted to get a batch of more items than there are in the queue."""


class NotEnoughSpace(Exception):
    """Raised if attempted to put a batch of more items than there is space for in the queue."""


def _calculate_memory_size(item_dtype: np.dtype, max_size: int) -> int:
    assert isinstance(item_dtype.itemsize, int)
    return (
        max_size * (item_dtype.itemsize + PRIORITY_DTYPE.itemsize + AGE_DTYPE.itemsize)
        + AGE_DTYPE.itemsize  # We add one more slot to keep the counter.
    )


# Simple random number generator.
# See: https://arxiv.org/pdf/2004.06278v2.pdf
# Here we use the same number for key and counter, which might make this a
# "bad" RNG, but it still seems pretty random when selecting over modest ranges.
@numba.njit(numba.types.uint64(numba.types.uint64, numba.types.int64))
def _simple_rng_inner(input: int, rounds: int) -> int:
    # Setup.
    counter = key = input
    y = x = counter * key
    z = y + key

    # Randomize.
    for round in range(rounds - 1):
        # Do a squaring plus counter addition.
        if round % 2 == 0:
            x = x * x + z
        else:
            x = x * x + y

        # Rotate the top 32 bits to the lower 32 bits for better randomness in
        # future rounds' squarings.
        x = (x >> 32) | (x << 32)

    # Finish.
    return x


def _simple_rng(input: int, rounds: int = 4) -> int:
    return _simple_rng_inner(input=input, rounds=rounds)


@numba.njit(
    numba.types.int64[:](numba.types.uint8[:], numba.types.int64, numba.types.int64)
)
def _find_k_open(priority_array: PriorityArray, k: int, start_idx: int):
    result = np.empty(shape=k, dtype=np.int64)
    search_idx = start_idx
    result_idx = 0
    while result_idx < k:
        if priority_array[search_idx] == 0:
            result[result_idx] = search_idx
            result_idx += 1
        search_idx = (search_idx + 1) % priority_array.shape[0]
    return result


@dataclass(frozen=True)
class SmallPriorityQueueSpec:
    name: str
    item_dtype: np.dtype
    max_size: int

    @classmethod
    def create(cls, name: str, item_dtype: DTypeLike, max_size: int):
        return cls(name=name, item_dtype=np.dtype(item_dtype), max_size=max_size)

    @property
    def lock_name(self) -> str:
        return f"/lodis_small_priorityqueue_semaphore_{self.name}"

    @property
    def memory_name(self) -> str:
        return f"lodis_small_priorityqueue_memory_{self.name}"


@dataclass(frozen=True)
class SmallPriorityQueue(Generic[T_ItemDtype, T_Item]):
    spec: SmallPriorityQueueSpec
    _lock: librt_semaphore.SemaphoreMemoryAddress
    _mem: SharedMemory
    _priority_array: PriorityArray
    _age_array: AgeArray
    _item_array: ItemArray

    def _is_open_slot_nolock(self) -> MaskArray:
        return self._priority_array == 0

    def qsize(self) -> int:
        """Return the size of the queue."""
        with librt_semaphore.acquire_context(self._lock):
            priority_array = self._priority_array
            return (priority_array > 0).astype(np.uint8).sum()

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        with librt_semaphore.acquire_context(self._lock):
            return bool(np.all(self._is_open_slot_nolock()))

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise."""
        with librt_semaphore.acquire_context(self._lock):
            return not bool(np.any(self._is_open_slot_nolock()))

    def get(self, block: bool = True) -> T_Item:
        """Remove and return a an item from the queue.

        If optional args 'block' is true, block if necessary until an item is available.
        """
        try:
            return self.get_batch(k=1, block=block)[0].copy()
        except NotEnoughItems:
            raise Empty()

    def put(
        self, item: T_Item, priority: int = LEAST_URGENT_PRIORITY, block: bool = True
    ) -> None:
        """Put an item into the queue.

        If optional args 'block' is true, block if necessary until space is available.
        """
        try:
            return self.put_batch(
                np.array([item], dtype=self.spec.item_dtype),  # type: ignore
                priority=priority,
                block=block,
            )
        except NotEnoughSpace:
            raise Full()

    async def put_async(
        self, item: T_Item, priority: int = LEAST_URGENT_PRIORITY
    ) -> None:
        """Put an item into the queue, waiting for space."""
        return await self.put_batch_async(
            np.array([item], dtype=self.spec.item_dtype),  # type: ignore
            priority=priority,
        )

    def get_nowait(self) -> T_Item:
        """Get an item from the queue.

        Raises the Empty exception if no item was available."""
        return self.get(block=False)

    def put_nowait(self, item: T_Item, priority: int = LEAST_URGENT_PRIORITY) -> None:
        """Put an item into the queue.

        Raises the Full exception if no space was available."""
        return self.put(item, priority, block=False)

    def get_batch(self, k: int, block: bool = True) -> ItemArray:
        """Remove and return a batch of items from the queue.

        If optional args 'block' is true, block if necessary until items are available.
        """
        if not block:
            return self.get_batch_nowait(k)
        else:
            # TODO: Consider something cleaner than polling.
            while True:
                try:
                    return self.get_batch_nowait(k)
                except NotEnoughItems:
                    sleep(POLL_SLEEP_DURATION)

    def put_batch(
        self,
        batch: Sequence[T_Item],
        priority: int = LEAST_URGENT_PRIORITY,
        block: bool = True,
    ) -> None:
        """Put a a batch of items into the queue.

        If optional args 'block' is true, block if necessary until space is available.
        """
        if not block:
            return self.put_batch_nowait(batch, priority)
        else:
            # TODO: Consider something cleaner than polling.
            while True:
                try:
                    return self.put_batch_nowait(batch, priority)
                except NotEnoughSpace:
                    sleep(POLL_SLEEP_DURATION)

    async def put_batch_async(
        self, batch: Sequence[T_Item], priority: int = LEAST_URGENT_PRIORITY
    ) -> None:
        """Put a a batch of items into the queue, waiting for space."""
        while True:
            try:
                return self.put_batch_nowait(batch, priority)
            except NotEnoughSpace:
                await asyncio.sleep(POLL_SLEEP_DURATION)

    def get_batch_nowait(self, k: int) -> ItemArray:
        with librt_semaphore.acquire_context(self._lock):
            # Check for not enough items.
            is_filled_slot = ~self._is_open_slot_nolock()
            if (is_filled_slot).astype(np.uint8).sum() < k:
                raise NotEnoughItems()

            priority_array = self._priority_array
            item_array = self._item_array
            age_array = self._age_array

            # Subset to examinging only nonempty items.
            filled_idx = np.nonzero(is_filled_slot)[0]
            present_priorities = priority_array[filled_idx]

            # Partition off the top K items.
            top_k_idx = np.argpartition(present_priorities, k - 1)[:k]
            np.argpartition(present_priorities, 0)

            # Break priority ties by age.
            kth_item_priority = present_priorities[top_k_idx[-1]]
            is_tied = present_priorities == kth_item_priority
            if is_tied.astype(np.uint8).sum() > 1:
                n_taken = is_tied[:k].astype(np.uint8).sum()
                tied_idx = np.nonzero(is_tied)[0]
                present_ages = age_array[filled_idx]
                tiebreak_winner_idx = tied_idx[
                    np.argpartition(present_ages[tied_idx], n_taken)[:n_taken]
                ]
                higher_than_tied_idx = top_k_idx[~is_tied[:k]]
                top_k_idx = np.concatenate([higher_than_tied_idx, tiebreak_winner_idx])

            # Move from indexing the present values to indexing all values.
            top_k_idx = filled_idx[top_k_idx]

            # Get selected items.
            result = item_array[top_k_idx].copy()

            # Mark as cleared.
            priority_array[top_k_idx] = 0

            del (
                priority_array,
                item_array,
                age_array,
                filled_idx,
                present_priorities,
                top_k_idx,
            )

            # Return.
            return result

    def put_batch_nowait(
        self, batch: Sequence[T_Item], priority: int = LEAST_URGENT_PRIORITY
    ) -> None:
        """Put an item into the queue.

        Raises the Full exception if no free slot was available.
        """
        assert MOST_URGENT_PRIORITY <= priority <= LEAST_URGENT_PRIORITY
        batch_array = np.array(batch, dtype=self.spec.item_dtype)
        k = batch_array.shape[0]
        with librt_semaphore.acquire_context(self._lock):
            # Check for not enough space.
            if (self._is_open_slot_nolock()).astype(np.uint8).sum() < k:
                raise NotEnoughSpace()

            priority_array = self._priority_array
            item_array = self._item_array
            age_array = self._age_array

            # Increment age counter.
            next_age = age_array[-1] + 1
            age_array[-1] = next_age

            # Find `k` open spots, starting in a random spot.
            start_idx = _simple_rng(next_age) % self.spec.max_size
            open_idx = _find_k_open(priority_array, k=k, start_idx=start_idx)

            # Insert away!
            item_array[open_idx] = batch_array
            priority_array[open_idx] = priority
            age_array[open_idx] = next_age


def allocate(spec: SmallPriorityQueueSpec) -> None:
    # Create a lock semaphore.
    librt_semaphore.allocate(spec.lock_name)

    # Create a SharedMemory, set it up, and close it.
    try:
        # Allocate memory.
        n_bytes_memory = _calculate_memory_size(
            item_dtype=spec.item_dtype, max_size=spec.max_size
        )
        mem = SharedMemory(name=spec.memory_name, create=True, size=n_bytes_memory)

        # Clear the memory.
        np.frombuffer(mem.buf, dtype=np.uint8)[:] = 0

        # Close things down.
        mem.close()
        del mem
    except Exception:
        librt_semaphore.unlink(spec.lock_name)
        raise


def unlink(spec: SmallPriorityQueueSpec) -> None:
    # Get the sems.
    lock = librt_semaphore.open(spec.lock_name)

    # Try to acquire the lock, but don't sweat it if we fail to.
    got_lock = librt_semaphore.acquire(lock, blocking=False)
    if not got_lock:
        warnings.warn(f"Unable to lock dictionary via semaphore")

    # Unlink the memory.
    memory = SharedMemory(spec.memory_name)
    memory.close()
    memory.unlink()
    del memory

    # Unlink the lock.
    librt_semaphore.close(lock)
    librt_semaphore.unlink(spec.lock_name)


def open(spec: SmallPriorityQueueSpec) -> SmallPriorityQueue:
    lock = librt_semaphore.open(name=spec.lock_name)
    mem = SharedMemory(name=spec.memory_name)
    # Tell Python to not be so clever about automatically deallocating this memory.
    # See: https://stackoverflow.com/a/73159334
    resource_tracker.unregister(name=f"/{mem.name}", rtype="shared_memory")
    end_priority_array = PRIORITY_DTYPE.itemsize * spec.max_size
    end_age_array = end_priority_array + AGE_DTYPE.itemsize * (spec.max_size + 1)
    # NOTE: We use `np.ndarray` directly to avoid buffer management that makes
    # closing this trickier. See: https://stackoverflow.com/a/72462697
    priority_array: PriorityArray = np.ndarray(
        shape=(spec.max_size,),
        dtype=PRIORITY_DTYPE,
        buffer=mem.buf[:end_priority_array],
    )
    age_array: AgeArray = np.ndarray(
        shape=(spec.max_size + 1,),
        dtype=AGE_DTYPE,
        buffer=mem.buf[end_priority_array:end_age_array],
    )
    item_array: ItemArray = np.ndarray(
        shape=(spec.max_size), dtype=spec.item_dtype, buffer=mem.buf[end_age_array:]
    )
    return SmallPriorityQueue(
        spec=spec,
        _lock=lock,
        _mem=mem,
        _priority_array=priority_array,
        _age_array=age_array,
        _item_array=item_array,
    )


def close(q: SmallPriorityQueue) -> None:
    librt_semaphore.close(q._lock)
    q._mem.close()


@contextlib.contextmanager
def open_ctx(spec: SmallPriorityQueueSpec) -> Iterator[SmallPriorityQueue]:
    q = open(spec)
    try:
        yield q
    finally:
        close(q)
