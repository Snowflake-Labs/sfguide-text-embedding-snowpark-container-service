import multiprocessing as mp
import signal
from multiprocessing.shared_memory import SharedMemory
from queue import Full

import numpy as np
import pytest
from lodis import hashmap
from lodis import librt_semaphore


def _ensure_not_allocated(name: str):
    fake_id = hashmap.HashMapSpec(name, np.dtype(np.bool_), 1)
    try:
        memory = SharedMemory(fake_id.memory_name)
        memory.close()
        memory.unlink()
        del memory
    except Exception:
        pass
    try:
        librt_semaphore.unlink(fake_id.lock_name)
    except Exception:
        pass


def test_allocate_and_unlink():
    _ensure_not_allocated("test_hashmap")
    hm_spec = hashmap.HashMapSpec.create(
        "test_hashmap", key_dtype="int64", value_dtype="S4", max_size=3
    )
    hashmap.allocate(hm_spec)
    hashmap.unlink(hm_spec)


def test_open_close():
    _ensure_not_allocated("test_hashmap")
    hm_spec = hashmap.HashMapSpec.create(
        "test_hashmap", key_dtype="int64", value_dtype="S4", max_size=3
    )
    hashmap.allocate(hm_spec)
    hm = hashmap.open(hm_spec)
    try:
        hashmap.close(hm)
    finally:
        hashmap.unlink(hm_spec)


def test_hm_functions():
    _ensure_not_allocated("test_hashmap")
    hm_spec = hashmap.HashMapSpec.create(
        "test_hashmap", key_dtype="int64", value_dtype="S4", max_size=3
    )
    hashmap.allocate(hm_spec)
    hm = hashmap.open(hm_spec)
    try:
        hm[1] = b"1234"
        hm[5] = b"abcd"
        assert hm[1] == b"1234"
        assert set(iter(hm)) == set(hm.keys()) == {1, 5}
        assert set(hm.values()) == {b"1234", b"abcd"}
        assert set(hm.items()) == {(1, b"1234"), (5, b"abcd")}
        del hm[1]
        with pytest.raises(KeyError):
            hm[1234]
        with pytest.raises(KeyError):
            del hm[1234]
        with pytest.raises(KeyError):
            hm[1]
    finally:
        hashmap.close(hm)
        hashmap.unlink(hm_spec)


def test_hm_full():
    _ensure_not_allocated("test_hashmap")
    hm_spec = hashmap.HashMapSpec.create(
        "test_hashmap", key_dtype="int64", value_dtype="S4", max_size=3
    )
    hashmap.allocate(hm_spec)
    hm = hashmap.open(hm_spec)
    try:
        hm[1] = b"1234"
        hm[5] = b"abcd"
        hm[0] = b"abcd"
        with pytest.raises(Full):
            hm[-55] = b"full"
        # Ensure we don't hit an infinite loop if we exhaustively linearly probe.
        assert 123 not in hm
        with pytest.raises(KeyError):
            hm[123]
        del hm[0]
        hm[-55] = b"okay"
    finally:
        hashmap.close(hm)
        hashmap.unlink(hm_spec)


def test_hm_collision():
    _ensure_not_allocated("test_hashmap")
    hm_spec = hashmap.HashMapSpec.create(
        "test_hashmap", key_dtype="int64", value_dtype="S4", max_size=3
    )
    hashmap.allocate(hm_spec)
    hm = hashmap.open(hm_spec)
    try:
        # hash(0) % 3 == hash(3) % 3 == 0
        hm[0] = b"1234"
        assert 3 not in hm
        with pytest.raises(KeyError):
            hm[3]
    finally:
        hashmap.close(hm)
        hashmap.unlink(hm_spec)


SHARED_PROC_MAP_SPEC = hashmap.HashMapSpec.create(
    name="test_hashmap", key_dtype="int64", value_dtype="S4", max_size=2
)


def _subproc_target():
    hm = hashmap.open(SHARED_PROC_MAP_SPEC)
    signal.signal(signal.SIGTERM, lambda *_: hashmap.close(hm))
    assert hm.pop_block(1) == b"0001"
    hm[2] = b"0002"
    hm[3] = b"0003"
    # This has to wait for the main process to pop something.
    hm.set_block(4, b"0004")


def test_two_processes():
    _ensure_not_allocated("test_hashmap")

    # Allocate a queue and spawn a separate process to do stuff with that queue.
    hashmap.allocate(SHARED_PROC_MAP_SPEC)
    proc = mp.get_context("spawn").Process(target=_subproc_target)
    proc.start()
    hm = hashmap.open(SHARED_PROC_MAP_SPEC)

    # Interact with the other process through the queue.
    try:
        hm[1] = b"0001"
        # This has to wait for the subprocess to insert something.
        assert hm.pop_block(3) == b"0003"
    finally:
        proc.terminate()
        proc.join()
        hashmap.close(hm)
        hashmap.unlink(SHARED_PROC_MAP_SPEC)


# TODO: Test `getitem_block` and `setitem_block`.
