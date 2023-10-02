import contextlib
import logging
import warnings
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from queue import Full
from time import sleep
from typing import cast
from typing import Generic
from typing import Iterator

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
FirstInLastInIdxArray = NDArray[np.int64]
FIRST_IN_LAST_IN_IDX_DTYPE = np.dtype(np.int64)


def _calculate_memory_size(item_dtype: np.dtype, max_size: int) -> int:
    assert isinstance(FIRST_IN_LAST_IN_IDX_DTYPE.itemsize, int) and isinstance(
        item_dtype.itemsize, int
    )
    return 2 * FIRST_IN_LAST_IN_IDX_DTYPE.itemsize + max_size * item_dtype.itemsize


@dataclass(frozen=True)
class QueueSpec:
    name: str
    item_dtype: np.dtype
    max_size: int

    @classmethod
    def create(cls, name: str, item_dtype: DTypeLike, max_size: int):
        return cls(name=name, item_dtype=np.dtype(item_dtype), max_size=max_size)

    @property
    def lock_name(self) -> str:
        return f"/lodis_queue_semaphore_{self.name}"

    @property
    def notempty_sem_name(self) -> str:
        return f"/lodis_queue_notempty_semaphore_{self.name}"

    @property
    def memory_name(self) -> str:
        return f"lodis_queue_memory_{self.name}"


def _calculate_notempty_qsize(
    first_in_idx: int, last_in_idx: int, max_size: int
) -> int:
    is_wraparound = first_in_idx > last_in_idx
    if is_wraparound:
        wraparound_amount = max_size - first_in_idx
        return last_in_idx + wraparound_amount + 1
    else:
        return last_in_idx - first_in_idx + 1


@dataclass(frozen=True)
class Queue(Generic[T_ItemDtype, T_Item]):
    spec: QueueSpec
    _lock: librt_semaphore.SemaphoreMemoryAddress
    _notempty_sem: librt_semaphore.SemaphoreMemoryAddress
    _mem: SharedMemory
    _first_in_last_in_idx_array: FirstInLastInIdxArray
    _item_array: ItemArray

    def _notempty(self) -> bool:
        """Checks for emptiness via the notempty semaphore.

        Should only be used when holding the lock sem and when *not* holding the
        notemtpy sem.
        """
        return librt_semaphore.getvalue(self._notempty_sem) > 0

    def qsize(self) -> int:
        """Return the size of the queue."""
        with librt_semaphore.acquire_context(self._lock):
            first_in_last_in_idx_array = self._first_in_last_in_idx_array
            is_empty = not self._notempty()
            first_in_idx, last_in_idx = first_in_last_in_idx_array
        if is_empty:
            # Last in should now be one lower (ahead of) first in.
            assert last_in_idx == (first_in_idx - 1) % self.spec.max_size
            return 0
        return _calculate_notempty_qsize(
            first_in_idx=first_in_idx,
            last_in_idx=last_in_idx,
            max_size=self.spec.max_size,
        )

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        with librt_semaphore.acquire_context(self._lock):
            return not self._notempty()

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise."""
        return self.qsize() == self.spec.max_size

    def get(self, block: bool = True) -> T_Item:
        """Remove and return an item from the queue.

        If optional args 'block' is true, block if necessary until an item is available.
        """
        if not block:
            return self.get_nowait()
        else:
            # TODO: Consider something cleaner than polling.
            while True:
                try:
                    return self.get_nowait()
                except Empty:
                    sleep(POLL_SLEEP_DURATION)

    def put(self, item: T_Item, block: bool = True) -> None:
        """Put an item into the queue.

        If optional args 'block' is true, block if necessary until space is available.
        """
        if not block:
            return self.put_nowait(item)
        else:
            # TODO: Consider something cleaner than polling.
            while True:
                try:
                    return self.put_nowait(item)
                except Full:
                    sleep(POLL_SLEEP_DURATION)

    def get_nowait(self) -> T_Item:
        """Get an item from the queue.

        Raises the Empty exception if no item was available."""
        with librt_semaphore.acquire_context(self._lock):
            # Check for empty.
            if not librt_semaphore.acquire(self._notempty_sem, blocking=False):
                raise Empty()

            first_in_last_in_idx_array = self._first_in_last_in_idx_array
            item_array = self._item_array

            # Look up the first-in item.
            first_in_idx, last_in_idx = first_in_last_in_idx_array
            item = cast(T_Item, item_array[first_in_idx])

            # Shift the first-in index.
            first_in_idx = (first_in_idx + 1) % self.spec.max_size
            first_in_last_in_idx_array[0] = first_in_idx

            # If we're now empty, we keep the notempty semaphore.
            # Otherwise we release it.
            # (We are empty if last in just got ahead of (lower than) first in.)
            # NOTE: We cannot use `_is_empty_nolock()` here because we're holding
            # the notempty semaphore that it checks.
            is_now_empty = last_in_idx == (first_in_idx - 1) % self.spec.max_size
            if not is_now_empty:
                librt_semaphore.release(self._notempty_sem)
            # logger.debug(
            #     f"QUEUE '{self.spec.name}' GET FINISHED: ITEM='{item}' | FI={first_in_idx} | LI={last_in_idx} | NOW_EMPTY = {is_now_empty} | NOTEMPTY = {self._notempty()} | QSIZE = {_calculate_notempty_qsize(first_in_idx, last_in_idx, self.spec.max_size)}"
            # )

            return item

    def put_nowait(self, item: T_Item) -> None:
        """Put an item into the queue.

        Raises the Full exception if no free slot was available.
        """
        with librt_semaphore.acquire_context(self._lock):
            first_in_last_in_idx_array = self._first_in_last_in_idx_array
            item_array = self._item_array

            # Get size/location info.
            is_empty = not self._notempty()
            first_in_idx, last_in_idx = first_in_last_in_idx_array

            # Check for full.
            is_full = (not is_empty) and (
                _calculate_notempty_qsize(
                    first_in_idx=first_in_idx,
                    last_in_idx=last_in_idx,
                    max_size=self.spec.max_size,
                )
                == self.spec.max_size
            )
            if is_full:
                raise Full()

            # Shift the last-in item by one.
            last_in_idx = (last_in_idx + 1) % self.spec.max_size
            first_in_last_in_idx_array[1] = last_in_idx

            # Insert the new last-in item.
            item_array[last_in_idx] = item

            # If we started empty, release the notempty sem.
            if is_empty:
                librt_semaphore.release(self._notempty_sem)
                assert self._notempty()
            # logger.debug(
            #     f"QUEUE '{self.spec.name}' PUT FINISHED: ITEM='{item}' FI={first_in_idx} | LI={last_in_idx} | WAS_EMPTY = {is_empty} | NOTEMPTY = {self._notempty()} | QSIZE = {_calculate_notempty_qsize(first_in_idx, last_in_idx, self.spec.max_size)}"
            # )


def allocate(spec: QueueSpec) -> None:
    # Create a lock semaphore.
    librt_semaphore.allocate(spec.lock_name)

    # Create, ACQUIRE, and then close a notempty semaphore.
    librt_semaphore.allocate(spec.notempty_sem_name)
    notempty_sem = librt_semaphore.open(spec.notempty_sem_name)
    librt_semaphore.acquire(notempty_sem)
    librt_semaphore.close(notempty_sem)

    # Create a SharedMemory, set it up, and close it.
    try:
        # Allocate memory.
        n_bytes_memory = _calculate_memory_size(
            item_dtype=spec.item_dtype, max_size=spec.max_size
        )
        mem = SharedMemory(name=spec.memory_name, create=True, size=n_bytes_memory)

        # Clear the memory.
        np.frombuffer(mem.buf, dtype=np.uint8)[:] = 0

        # Set the first-in index to 1.
        first_in_last_in_idx_array = np.frombuffer(
            mem.buf[: 2 * FIRST_IN_LAST_IN_IDX_DTYPE.itemsize],
            dtype=FIRST_IN_LAST_IN_IDX_DTYPE,
        )
        first_in_last_in_idx_array[0] = 1

        # Close things down.
        del first_in_last_in_idx_array
        mem.close()
        del mem
    except Exception:
        librt_semaphore.unlink(spec.lock_name)
        librt_semaphore.unlink(spec.notempty_sem_name)
        raise


def unlink(spec: QueueSpec) -> None:
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

    # Unlink the notempty sem.
    librt_semaphore.unlink(spec.notempty_sem_name)

    # Unlink the lock.
    librt_semaphore.close(lock)
    librt_semaphore.unlink(spec.lock_name)


def open(spec: QueueSpec) -> Queue:
    lock = librt_semaphore.open(name=spec.lock_name)
    notempty_sem = librt_semaphore.open(name=spec.notempty_sem_name)
    mem = SharedMemory(name=spec.memory_name)
    # Tell Python to not be so clever about automatically deallocating this memory.
    # See: https://stackoverflow.com/a/73159334
    resource_tracker.unregister(name=f"/{mem.name}", rtype="shared_memory")

    # NOTE: We use `np.ndarray` directly to avoid buffer management that makes
    # closing this trickier. See: https://stackoverflow.com/a/72462697

    first_in_last_in_idx_array: FirstInLastInIdxArray = np.ndarray(
        shape=(2,),
        dtype=FIRST_IN_LAST_IN_IDX_DTYPE,
        buffer=mem.buf[: 2 * FIRST_IN_LAST_IN_IDX_DTYPE.itemsize],
    )
    item_array: ItemArray = np.ndarray(
        shape=(spec.max_size,),
        dtype=spec.item_dtype,
        buffer=mem.buf[2 * FIRST_IN_LAST_IN_IDX_DTYPE.itemsize :],
    )
    return Queue(
        spec=spec,
        _lock=lock,
        _notempty_sem=notempty_sem,
        _mem=mem,
        _first_in_last_in_idx_array=first_in_last_in_idx_array,
        _item_array=item_array,
    )


def close(q: Queue) -> None:
    librt_semaphore.close(q._lock)
    librt_semaphore.close(q._notempty_sem)
    # Clean up all references to the memory buffer before we close it.
    # (This is enforced by `SharedMemory`, it's not just good practice!)
    object.__delattr__(q, "_first_in_last_in_idx_array")
    object.__delattr__(q, "_item_array")
    q._mem.close()


@contextlib.contextmanager
def open_ctx(spec: QueueSpec) -> Iterator[Queue]:
    q = open(spec)
    try:
        yield q
    finally:
        close(q)
