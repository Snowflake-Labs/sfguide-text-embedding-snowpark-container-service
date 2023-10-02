from multiprocessing.shared_memory import SharedMemory

import numpy as np
from lodis import librt_semaphore
from lodis import queue
from lodis import small_priority_queue


def _ensure_semaphore_not_allocated(name: str):
    try:
        librt_semaphore.unlink(name)
    except Exception:
        pass


def _ensure_queue_not_allocated(name: str):
    fake_id = queue.QueueSpec(name, np.dtype(np.bool_), 1)
    try:
        memory = SharedMemory(fake_id.memory_name)
        memory.close()
        memory.unlink()
        del memory
    except Exception:
        pass
    try:
        librt_semaphore.unlink(fake_id.lock_name)
        librt_semaphore.unlink(fake_id.notempty_sem_name)
    except Exception:
        pass


def _ensure_small_priority_queue_not_allocated(name: str):
    fake_id = small_priority_queue.SmallPriorityQueueSpec(name, np.dtype(np.bool_), 1)
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
