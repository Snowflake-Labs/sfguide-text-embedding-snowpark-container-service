"""Logic for managing the embedding loop as a subprocess."""
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

_embedding_process: Optional[subprocess.Popen] = None

ROOT_DIR = Path(__file__).resolve().parents[1]
EMBEDDING_ENV_PYTHON_EXECUTABLE_PATH = ROOT_DIR / "venv_embed_loop" / "bin" / "python"
EMBEDDING_LOOP_MAIN_PATH = ROOT_DIR / "service_embed_loop" / "embedding_worker.py"


def startup_embedding_process() -> subprocess.Popen:
    global _embedding_process
    assert _embedding_process is None
    if not (
        EMBEDDING_ENV_PYTHON_EXECUTABLE_PATH.is_file()
        and os.access(EMBEDDING_ENV_PYTHON_EXECUTABLE_PATH, os.X_OK)
    ):
        raise RuntimeError(
            f'No executable found for the embedding process at path "{EMBEDDING_ENV_PYTHON_EXECUTABLE_PATH}"'
        )
    if not EMBEDDING_LOOP_MAIN_PATH.is_file():
        raise RuntimeError(
            f'No embedding loop program file found at path "{EMBEDDING_LOOP_MAIN_PATH}"'
        )
    _embedding_process = subprocess.Popen(
        [str(EMBEDDING_ENV_PYTHON_EXECUTABLE_PATH), str(EMBEDDING_LOOP_MAIN_PATH)],
        stdout=subprocess.DEVNULL,
    )
    return _embedding_process


def shutdown_embedding_process(total_timeout: float = 10) -> None:
    global _embedding_process
    assert isinstance(_embedding_process, subprocess.Popen)
    _embedding_process.terminate()
    try:
        _embedding_process.wait(timeout=total_timeout / 2)
    except subprocess.TimeoutExpired:
        _embedding_process.kill()
        try:
            _embedding_process.wait(timeout=total_timeout / 2)
        except subprocess.TimeoutExpired:
            logging.getLogger(__name__).warning(
                f"Tried killing embedding subprocess (PID {_embedding_process.pid}) "
                f"but gave up after {total_timeout} seconds"
            )
