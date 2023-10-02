import numpy as np
import pytest
from _utils import _ensure_small_priority_queue_not_allocated
from lodis import small_priority_queue


def test_batches():
    _ensure_small_priority_queue_not_allocated("test_queue")
    q_spec = small_priority_queue.SmallPriorityQueueSpec.create(
        name="test_queue", item_dtype="S4", max_size=3
    )
    small_priority_queue.allocate(q_spec)
    q = small_priority_queue.open(q_spec)
    try:
        pass
    finally:
        small_priority_queue.close(q)
        small_priority_queue.unlink(q_spec)


def test_priority():
    _ensure_small_priority_queue_not_allocated("test_queue")
    q_spec = small_priority_queue.SmallPriorityQueueSpec.create(
        name="test_queue", item_dtype="S4", max_size=10
    )
    small_priority_queue.allocate(q_spec)
    q = small_priority_queue.open(q_spec)
    try:
        # Check boundary conditions on priority.
        with pytest.raises(AssertionError):
            q.put(b"asdf", priority=256)
        with pytest.raises(AssertionError):
            q.put_nowait(b"asdf", priority=12345)
        with pytest.raises(AssertionError):
            q.put_batch([b"asdf"], priority=0)
        with pytest.raises(AssertionError):
            q.put(b"asdf", priority=0)

        # Check priority.
        q.put(b"0006")
        q.put_batch_nowait([b"0004", b"0005"], priority=40)
        q.put(b"0002", priority=20)
        q.put(b"0001", priority=10)
        q.put(b"0003", priority=30)
        assert q.get(block=False) == b"0001"
        assert q.get(block=False) == b"0002"
        assert q.get(block=False) == b"0003"
        assert q.get(block=False) == b"0004"
        assert q.get(block=False) == b"0005"
        assert q.get(block=False) == b"0006"

    finally:
        small_priority_queue.close(q)
        small_priority_queue.unlink(q_spec)


SHARED_PROC_QUEUE_SPEC_INFO = dict(name="test_queue", item_dtype=np.int64, max_size=2)
SHARED_PROC_WAIT_SEM_NAME = "/test_sem"

# TODO: Test the _nowait() variants.
