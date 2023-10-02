import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from time import sleep

import multiprocess_logging as mpl
import numpy as np
from lodis import librt_semaphore
from lodis import queue

QUEUE_NAME = "test_log_queue"
COMMUNICATION_LOCK_NAME = "/test_sync_lock"
LOG_QUEUE_SPEC = queue.QueueSpec.create(
    name=QUEUE_NAME, item_dtype=np.dtype("<U1000"), max_size=10
)


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


def _ensure_semaphore_not_allocated(name: str):
    try:
        librt_semaphore.unlink(name)
    except Exception:
        pass


def _subproc_target():
    # Use the multiprocess logging handler.
    log_handler = mpl.LodisQueueLogHandler(LOG_QUEUE_SPEC)
    logger = logging.getLogger(__name__)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)

    # Send some log messages.
    logger.debug("Debug")
    logger.info("Info")
    logger.warning("Warning")
    logger.critical("Critical")
    try:
        raise RuntimeError("Runtime Error")
    except Exception:
        logger.exception("Exception")

    # Close the handler.
    log_handler.close()

    # Tell the main process it can stop waiting for me by releasing the lock.
    lock = librt_semaphore.open(COMMUNICATION_LOCK_NAME)
    librt_semaphore.release(lock)
    librt_semaphore.close(lock)


def test_two_processes(caplog):
    # Create the lodis queue that will pass log messages between processes.
    _ensure_queue_not_allocated(QUEUE_NAME)
    queue.allocate(LOG_QUEUE_SPEC)

    # Create a lock item for process synchronization.
    _ensure_semaphore_not_allocated(COMMUNICATION_LOCK_NAME)
    librt_semaphore.allocate(COMMUNICATION_LOCK_NAME)
    lock = librt_semaphore.open(COMMUNICATION_LOCK_NAME)
    assert librt_semaphore.acquire(lock, blocking=False)

    # Prepare to receive log messages over the queue.
    mpl.start_listening_for_logs(LOG_QUEUE_SPEC)

    with caplog.at_level(logging.DEBUG):
        # Allocate spawn a separate process to send log messages back to us.
        proc = mp.get_context("spawn").Process(target=_subproc_target)
        proc.start()

        # Wait for the other process to do its thing.
        librt_semaphore.acquire(lock, blocking=True)

        # Wait for the queue to empty via the log listener thread.
        for _retry in range(500):
            if not mpl._logging_queue.empty():
                sleep(0.01)
            else:
                break

        # Ensure the logs went through.
        try:
            assert "Debug" not in caplog.text
            assert "Info" in caplog.text
            assert "Warning" in caplog.text
            assert "Critical" in caplog.text
            assert "Exception" in caplog.text
        finally:
            proc.terminate()
            proc.join()
            mpl.stop_listening_for_logs()
            queue.unlink(LOG_QUEUE_SPEC)
            librt_semaphore.unlink(COMMUNICATION_LOCK_NAME)
