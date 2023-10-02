import multiprocessing as mp
import signal
from queue import Empty
from queue import Full
from types import ModuleType
from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import numpy as np
import pytest
from _utils import _ensure_queue_not_allocated
from _utils import _ensure_semaphore_not_allocated
from _utils import _ensure_small_priority_queue_not_allocated
from lodis import librt_semaphore
from lodis import queue
from lodis import small_priority_queue


def _ensure_no_queue_allocated(name: str):
    _ensure_queue_not_allocated(name)
    _ensure_small_priority_queue_not_allocated(name)


parameterized_on_both_queues = pytest.mark.parametrize(
    argnames=["module", "spec"],
    argvalues=[
        (queue, queue.QueueSpec),
        (small_priority_queue, small_priority_queue.SmallPriorityQueueSpec),
    ],
    ids=["Queue", "SmallPriorityQueue"],
)


@parameterized_on_both_queues
def test_allocate_and_unlink(module, spec):
    _ensure_no_queue_allocated("test_queue")
    q_spec = spec.create(name="test_queue", item_dtype="int64", max_size=3)
    module.allocate(q_spec)
    module.unlink(q_spec)


@parameterized_on_both_queues
def test_open_close(module, spec):
    _ensure_no_queue_allocated("test_queue")
    q_spec = spec.create(name="test_queue", item_dtype="S4", max_size=3)
    module.allocate(q_spec)
    q = module.open(q_spec)
    try:
        module.close(q)
    finally:
        module.unlink(q_spec)


@parameterized_on_both_queues
def test_use_simple(module, spec):
    _ensure_no_queue_allocated("test_queue")
    q_spec = spec.create(name="test_queue", item_dtype="S4", max_size=3)
    module.allocate(q_spec)
    q = module.open(q_spec)
    try:
        # Put some items.
        assert q.qsize() == 0
        assert q.empty()
        assert not q.full()

        q.put(b"0001")
        assert q.qsize() == 1
        assert not q.empty()
        assert not q.full()

        assert q.get(block=False) == b"0001"
        assert q.qsize() == 0
        assert q.empty()
        assert not q.full()

    finally:
        module.close(q)
        module.unlink(q_spec)


@parameterized_on_both_queues
def test_use(module, spec):
    _ensure_no_queue_allocated("test_queue")
    q_spec = spec.create(name="test_queue", item_dtype="S4", max_size=3)
    module.allocate(q_spec)
    q = module.open(q_spec)
    try:
        # Put some items.
        assert q.qsize() == 0
        assert q.empty()
        assert not q.full()

        q.put(b"0001")
        assert q.qsize() == 1
        assert not q.empty()
        assert not q.full()

        q.put(b"0002")
        assert q.qsize() == 2
        assert not q.empty()
        assert not q.full()

        q.put(b"0003")
        assert q.qsize() == 3
        assert not q.empty()
        assert q.full()

        with pytest.raises(Full):
            q.put(b"0004", block=False)
        assert q.qsize() == 3
        assert not q.empty()
        assert q.full()

        # Get and refill.
        assert q.get(block=False) == b"0001"
        assert q.qsize() == 2
        assert not q.empty()
        assert not q.full()

        assert q.get(block=True) == b"0002"
        assert q.qsize() == 1
        assert not q.empty()
        assert not q.full()

        q.put(b"0004")
        assert q.qsize() == 2
        assert not q.empty()
        assert not q.full()

        q.put(b"0005")
        assert q.qsize() == 3
        assert not q.empty()
        assert q.full()

        # Get without refilling.
        assert q.get(block=False) == b"0003"
        assert q.get(block=False) == b"0004"
        assert q.get(block=False) == b"0005"
        assert q.qsize() == 0
        assert q.empty()
        assert not q.full()

        with pytest.raises(Empty):
            q.get(block=False)
        assert q.qsize() == 0
        assert q.empty()
        assert not q.full()

    finally:
        module.close(q)
        module.unlink(q_spec)


SHARED_PROC_QUEUE_SPEC_INFO: Dict[str, Any] = dict(
    name="test_queue", item_dtype=np.int64, max_size=2
)
SHARED_PROC_WAIT_SEM_NAME = "/test_sem"


def _subproc_target(queue_kind_name: str):
    module: ModuleType
    spec: Type[Union[queue.QueueSpec, small_priority_queue.SmallPriorityQueueSpec]]

    if queue_kind_name == "queue":
        module, spec = queue, queue.QueueSpec
    elif queue_kind_name == "small_priority_queue":
        module, spec = small_priority_queue, small_priority_queue.SmallPriorityQueueSpec
    else:
        raise ValueError(queue_kind_name)
    q = module.open(spec.create(**SHARED_PROC_QUEUE_SPEC_INFO))
    signal.signal(signal.SIGTERM, lambda *_: module.close(q))
    assert q.get(block=True) == 1
    q.put(2)

    # Signal we're done by releasing the semaphore that the main process waits on.
    sem = librt_semaphore.open(SHARED_PROC_WAIT_SEM_NAME)
    librt_semaphore.release(sem)
    librt_semaphore.release(sem)
    librt_semaphore.close(sem)

    q.put(3)
    q.put(4)


@parameterized_on_both_queues
def test_two_processes(module, spec):
    _ensure_no_queue_allocated("test_queue")
    _ensure_semaphore_not_allocated(SHARED_PROC_WAIT_SEM_NAME)

    # Create a locked semaphore that this process will wait on, so that the
    # other process can signal when it's finished with its thing.
    librt_semaphore.allocate(SHARED_PROC_WAIT_SEM_NAME)
    wait_sem = librt_semaphore.open(SHARED_PROC_WAIT_SEM_NAME)
    librt_semaphore.acquire(wait_sem)

    # Allocate a queue and spawn a separate process to do stuff with that queue.
    queue_spec = spec.create(**SHARED_PROC_QUEUE_SPEC_INFO)
    module.allocate(queue_spec)
    proc = mp.get_context("spawn").Process(
        target=_subproc_target, args=(module.__name__.split(".")[-1],)
    )
    proc.start()
    q = module.open(queue_spec)

    # Interact with the other process through the queue.
    try:
        q.put(1)
        # Wait for the other process to tell us its ready.
        librt_semaphore.acquire(wait_sem)
        assert q.get() == 2
        assert q.get() == 3
        assert q.get() == 4
    finally:
        proc.terminate()
        proc.join()
        module.close(q)
        module.unlink(queue_spec)
        librt_semaphore.close(wait_sem)
        librt_semaphore.unlink(SHARED_PROC_WAIT_SEM_NAME)


# TODO: Test `put_nowait` and `get_nowait`.
