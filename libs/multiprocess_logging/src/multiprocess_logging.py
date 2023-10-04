"""Enables the embedding loop to pass its logs to the API via a `lodis` queue."""
import logging
import threading
from typing import Optional

import lodis.hashmap
import lodis.queue

END_LOGGING_SENTINEL_MESSAGE = "__END_LOGGING_SENTINEL__"


# Global logging state.
_logging_queue: Optional[lodis.queue.Queue] = None
_logging_thread: Optional[threading.Thread] = None


class LodisQueueLogHandler(logging.Handler):
    """
    This handler sends events to a lodis queue. Some record fields are lost.
    """

    def __init__(self, queue_spec: lodis.queue.QueueSpec):
        logging.Handler.__init__(self)
        self.queue = lodis.queue.open(queue_spec)

    def emit(self, record):
        try:
            self.queue.put_nowait(self.format(record))  # type: ignore
        except Exception:
            pass

    def close(self):
        lodis.queue.close(self.queue)


def _lodis_queue_log_listen(log_queue: lodis.queue.Queue):
    while True:
        message = log_queue.get(block=True)
        if message == END_LOGGING_SENTINEL_MESSAGE:
            break
        record = logging.LogRecord(
            name="embedding_loop_subprocess",
            level=logging.INFO,
            pathname="",
            lineno=-1,
            msg=message,
            args=None,
            exc_info=None,
        )

        logging.getLogger(record.name).handle(record)


def start_listening_for_logs(log_queue_spec: lodis.queue.QueueSpec):
    """Set up a lodis logging queue so that non-main processes can send logs to
    the main process.
    """
    global _logging_thread, _logging_queue
    assert _logging_queue is None
    assert _logging_thread is None
    _logging_queue = lodis.queue.open(log_queue_spec)
    _logging_thread = threading.Thread(
        target=_lodis_queue_log_listen, args=(_logging_queue,)
    )
    _logging_thread.start()


def stop_listening_for_logs():
    """Gracefully stops the listener thread started by `start_listening_for_logs`."""
    global _logging_thread, _logging_queue
    assert isinstance(_logging_queue, lodis.queue.Queue)
    assert isinstance(_logging_thread, threading.Thread)
    _logging_queue.put(END_LOGGING_SENTINEL_MESSAGE)
    _logging_thread.join()
    lodis.queue.close(_logging_queue)


def setup_root_handler(log_handler: logging.Handler, level: int = logging.INFO) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(level)
