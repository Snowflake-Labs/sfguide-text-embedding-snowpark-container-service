import logging

import lodis.hashmap
import lodis.queue
import lodis.small_priority_queue
import multiprocess_logging
import uvicorn
from starlette.applications import Starlette

from service_api import api
from service_api import embed_loop_process_manager
from services_common_code import config

logger = logging.getLogger(__name__)

LOGGING_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"


def main():
    app = Starlette(routes=api.ROUTES, on_startup=[_setup], on_shutdown=[_teardown])
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=logging.INFO)


def _setup():
    logger.info("Beginning application setup")
    logger.info("Allocating shared-memory data structures")
    lodis.small_priority_queue.allocate(config.INPUT_QUEUE_SPEC)
    lodis.hashmap.allocate(config.RESULT_MAP_SPEC)
    lodis.queue.allocate(config.LOG_QUEUE_SPEC)
    logger.info("Launching log listener thread")
    multiprocess_logging.start_listening_for_logs(config.LOG_QUEUE_SPEC)
    logger.info("Spawning the embed loop process")
    embed_loop_process_manager.startup_embedding_process()
    logger.info("Preparing API state")
    api.setup_state(
        embedding_input_queue_spec=config.INPUT_QUEUE_SPEC,
        embedding_result_map_spec=config.RESULT_MAP_SPEC,
    )
    logger.info("Application setup complete")


def _teardown():
    logger.info("Beginning application teardown")
    embed_loop_process_manager.shutdown_embedding_process()
    multiprocess_logging.stop_listening_for_logs()
    api.teardown_state()
    lodis.queue.unlink(config.LOG_QUEUE_SPEC)
    lodis.small_priority_queue.unlink(config.INPUT_QUEUE_SPEC)
    lodis.hashmap.unlink(config.RESULT_MAP_SPEC)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    main()
