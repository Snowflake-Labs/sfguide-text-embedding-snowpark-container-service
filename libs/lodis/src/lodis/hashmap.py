import asyncio
import contextlib
import warnings
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from queue import Full
from time import sleep
from typing import cast
from typing import Dict
from typing import ItemsView
from typing import Iterator
from typing import KeysView
from typing import MutableMapping
from typing import ValuesView

import numpy as np
from lodis import librt_semaphore
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from ._constants import POLL_SLEEP_DURATION
from ._typing import NonscalarNDArray
from ._typing import T_Key
from ._typing import T_KVDtype
from ._typing import T_Value
from ._utils import unregister_shared_memory_tracking

# Type declarations.
KVArray = NonscalarNDArray[T_KVDtype]
SizeArray = NDArray[np.int64]
SIZE_DTYPE = np.dtype(np.int64)


def _kvdtype(key_dtype: np.dtype, value_dtype: np.dtype) -> np.dtype:
    return np.dtype(
        [("is_taken", np.bool_), ("key", key_dtype), ("value", value_dtype)]
    )


def _calculate_memory_size(kvdtype: np.dtype, max_size: int) -> int:
    assert isinstance(SIZE_DTYPE.itemsize, int) and isinstance(kvdtype.itemsize, int)
    return SIZE_DTYPE.itemsize + max_size * kvdtype.itemsize


@dataclass(frozen=True)
class HashMapSpec:
    name: str
    kvdtype: np.dtype
    max_size: int

    @classmethod
    def create(
        cls, name: str, key_dtype: DTypeLike, value_dtype: DTypeLike, max_size: int
    ):
        kvdtype = _kvdtype(np.dtype(key_dtype), np.dtype(value_dtype))
        return cls(name=name, kvdtype=kvdtype, max_size=max_size)

    @property
    def lock_name(self) -> str:
        return f"/lodis_hashmap_semaphore_{self.name}"

    @property
    def memory_name(self) -> str:
        return f"lodis_hashmap_memory_{self.name}"

    @property
    def key_dtype(self) -> np.dtype:
        return self.kvdtype["key"]

    @property
    def value_dtype(self) -> np.dtype:
        return self.kvdtype["key"]


def _del_scoot_collisions(kvarray: KVArray, deleted_pos: int) -> None:
    """Scoot any collisions coming after the deleted item back one slot."""
    prev_pos = deleted_pos
    check_pos = (prev_pos + 1) % kvarray.shape[0]

    # Check until we hit an empty slot.
    while kvarray[check_pos]["is_taken"]:
        # Check if a collision happened, and if so, scoot to un-collide.
        intended_pos = hash(kvarray[check_pos]["key"]) % kvarray.shape[0]
        is_collision_in_order = intended_pos <= prev_pos <= check_pos
        is_collision_looped_1 = check_pos < intended_pos <= prev_pos
        is_collision_looped_2 = prev_pos <= check_pos < intended_pos
        if is_collision_in_order or is_collision_looped_1 or is_collision_looped_2:
            # Scoot by copying back once, then clearing the check pos.
            kvarray[prev_pos] = kvarray[check_pos]
            kvarray[check_pos : check_pos + 1].view(np.uint8)[:] = 0

            # Continue checking for a scoot triggered by this scoot.
            prev_pos = check_pos
        # Check the next position.
        check_pos = (check_pos + 1) % kvarray.shape[0]


@dataclass(frozen=True)
class HashMap(MutableMapping[T_Key, T_Value]):
    spec: HashMapSpec
    _lock: librt_semaphore.SemaphoreMemoryAddress
    _mem: SharedMemory
    _size_array: SizeArray
    _kvarray: KVArray

    def _cast_key(self, key: object) -> T_Key:
        # Tries to get the key cast to the key numpy type.
        # (e.g. cast to int64 instead of Python int)
        try:
            return cast(T_Key, np.array(key, dtype=self.spec.key_dtype)[()])
        except Exception as e:
            raise TypeError(
                f"Provided key type {type(key)} cannot be cast to map key type {self.spec.key_dtype}"
            ) from e

    def _idx_of_nolock(self, key: T_Key) -> int:
        """Finds the index of a key, returning -1 if the key is absent."""
        kvarray = self._kvarray
        n_slots = kvarray.shape[0]
        initial_pos = hash(key) % n_slots
        pos = initial_pos
        while True:
            if not kvarray[pos]["is_taken"]:
                return -1
            else:
                # If we've found the key, return the position.
                if kvarray[pos]["key"] == key:
                    return pos

                # Otherwise, move on to the next slot.
                pos = (pos + 1) % n_slots

                # Avoid infinite loop in edge case of full kvarray and key not in hm.
                if pos == initial_pos:
                    return -1

    def _set_nolock(self, key: T_Key, value: T_Value) -> None:
        # Hash and resolve collisions.
        kvarray, size_array = self._kvarray, self._size_array
        n_slots = kvarray.shape[0]
        initial_pos = hash(key) % n_slots
        pos = initial_pos
        while True:
            # Case 1: Empty slot, so key not in dict, so increment dict size
            # before setting the entry.
            if not kvarray[pos]["is_taken"]:
                np.add.at(size_array, (), 1)
                kvarray[pos] = (True, key, value)
                return
            # Case 2: Matching slot, so key in dict, so update the entry.
            elif kvarray[pos]["key"] == key:
                kvarray[pos]["value"] = value
                return
            # Case 3: Collision, check for full and then try the next slot.
            else:
                pos = (pos + 1) % n_slots
                # If we circle back around the entier kvarray, we've run out of room.
                if pos == initial_pos:
                    raise Full()

    def _del_nolock(self, key: T_Key) -> None:
        kvarray, size_array = self._kvarray, self._size_array

        # Get the position to delete.
        pos = self._idx_of_nolock(key)
        if pos == -1:
            raise KeyError(key)

        # Perform the delete.
        np.subtract.at(size_array, (), 1)
        kvarray[pos : pos + 1].view(np.uint8)[:] = 0
        _del_scoot_collisions(kvarray, pos)

    def _get_nolock(self, key: T_Key) -> T_Value:
        pos = self._idx_of_nolock(key)
        if pos == -1:
            raise KeyError(key)
        value = self._kvarray[pos]["value"]
        # NOTE: We must copy since the data type could be such that numpy
        # returns a view, and that breaks things like `.pop()`.
        return value.copy()

    def __getitem__(self, key: T_Key) -> T_Value:
        key = self._cast_key(key)
        with librt_semaphore.acquire_context(self._lock):
            return self._get_nolock(key)

    def __setitem__(self, key: T_Key, value: T_Value) -> None:
        key = self._cast_key(key)
        with librt_semaphore.acquire_context(self._lock):
            return self._set_nolock(key, value)

    def pop_block(self, key: T_Key) -> T_Value:
        """Similar to `Queue.get(block=True)` in semantics, this method polls,
        retrying until a `KeyError` exception is not raised."""
        while True:
            try:
                return self.pop(key)
            except KeyError:
                sleep(POLL_SLEEP_DURATION)

    async def pop_block_async(self, key: T_Key) -> T_Value:
        """Async equivalent of `pop_block`."""
        while True:
            try:
                return self.pop(key)
            except KeyError:
                await asyncio.sleep(POLL_SLEEP_DURATION)

    def set_block(self, key: T_Key, value: T_Value) -> None:
        """Similar to `Queue.put(block=True)` in semantics, this method polls,
        retrying until a `Full` exception is not raised."""
        while True:
            try:
                self[key] = value
                return
            except Full:
                sleep(POLL_SLEEP_DURATION)

    async def set_block_async(self, key: T_Key, value: T_Value) -> None:
        """Async equivalent of `set_block`."""
        while True:
            try:
                self[key] = value
                return
            except Full:
                await asyncio.sleep(POLL_SLEEP_DURATION)

    def __delitem__(self, key: T_Key) -> None:
        key = self._cast_key(key)
        with librt_semaphore.acquire_context(self._lock):
            return self._del_nolock(key)

    def __len__(self) -> int:
        with librt_semaphore.acquire_context(self._lock):
            return self._size_array.item()

    def __iter__(self) -> Iterator[T_Key]:
        with librt_semaphore.acquire_context(self._lock):
            kvarray = self._kvarray
            items = kvarray["key"][kvarray["is_taken"]]
        yield from items

    def asdict(self) -> Dict[T_Key, T_Value]:
        with librt_semaphore.acquire_context(self._lock):
            kvarray = self._kvarray
            mask = kvarray["is_taken"]
            keys = kvarray["key"][mask]
            values = kvarray["value"][mask]
        return dict(zip(keys, values))

    # Optimized `pop` override: Take the lock only once.
    _pop_default_sentinel = cast(T_Value, object())

    def pop(self, key: T_Key, /, default: T_Value = _pop_default_sentinel) -> T_Value:  # type: ignore # The MutableMapping type annotation is really weird here!
        """D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
        If key is not found, d is returned if given, otherwise KeyError is raised.
        """
        cast_key = self._cast_key(key)
        with librt_semaphore.acquire_context(self._lock):
            try:
                result = self._get_nolock(cast_key)
            except KeyError:
                if default is not self._pop_default_sentinel:
                    return default
                raise
            self._del_nolock(cast_key)
            return result

    # TODO: Optimize performance.
    def keys(self) -> KeysView[T_Key]:
        return self.asdict().keys()

    # TODO: Optimize performance.
    def values(self) -> ValuesView[T_Value]:
        return self.asdict().values()

    # TODO: Optimize performance.
    def items(self) -> ItemsView[T_Key, T_Value]:
        return self.asdict().items()

    def __contains__(self, key: object) -> bool:
        with librt_semaphore.acquire_context(self._lock):
            return self._idx_of_nolock(self._cast_key(key)) != -1

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, HashMap):
            return False
        with librt_semaphore.acquire_context(self._lock):
            with librt_semaphore.acquire_context(__value._lock):
                return bytes(self._mem.buf) == bytes(__value._mem.buf)

    def __ne__(self, __value: object) -> bool:
        return not (self == __value)

    def clear(self) -> None:
        with librt_semaphore.acquire_context(self._lock):
            np.frombuffer(self._mem.buf, dtype=np.uint8)[:] = 0


def allocate(spec: HashMapSpec) -> None:
    # Create a lock.
    librt_semaphore.allocate(spec.lock_name)

    # Create a SharedMemory, clear it, and close it.
    try:
        n_bytes_memory = _calculate_memory_size(
            kvdtype=spec.kvdtype, max_size=spec.max_size
        )
        mem = SharedMemory(name=spec.memory_name, create=True, size=n_bytes_memory)
        np.frombuffer(mem.buf, dtype=np.uint8)[:] = 0
        mem.close()
        del mem
    except Exception:
        librt_semaphore.unlink(spec.lock_name)
        raise


def unlink(spec: HashMapSpec) -> None:
    # Get the lock.
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


def open(spec: HashMapSpec) -> HashMap:
    lock = librt_semaphore.open(name=spec.lock_name)
    mem = SharedMemory(name=spec.memory_name)
    unregister_shared_memory_tracking(mem.name)
    # NOTE: We use `np.ndarray` directly to avoid buffer management that makes
    # closing this trickier. See: https://stackoverflow.com/a/72462697
    size_array: SizeArray = np.ndarray(
        shape=(1,), dtype=SIZE_DTYPE, buffer=mem.buf[: SIZE_DTYPE.itemsize]
    )
    kvarray: KVArray = np.ndarray(
        shape=(spec.max_size,),
        dtype=spec.kvdtype,
        buffer=mem.buf[SIZE_DTYPE.itemsize :],
    )
    return HashMap(
        spec=spec, _lock=lock, _mem=mem, _size_array=size_array, _kvarray=kvarray
    )


def close(hm: HashMap) -> None:
    librt_semaphore.close(hm._lock)
    # Clean up all references to the memory buffer before we close it.
    # (This is enforced by `SharedMemory`, it's not just good practice!)
    object.__delattr__(hm, "_size_array")
    object.__delattr__(hm, "_kvarray")
    hm._mem.close()


@contextlib.contextmanager
def open_ctx(spec: HashMapSpec) -> Iterator[HashMap]:
    hm = open(spec)
    try:
        yield hm
    finally:
        close(hm)
