from pathlib import Path
from typing import cast
from typing import Iterator

from python_on_whales import docker

DOCKERFILE_PATH = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile"
CONTEXT_PATH = Path(__file__).resolve().parents[3]


def build() -> None:
    """Usage: `build_embed_service --dockerfile-path services/e5_service/deploy/Dockerfile .`"""
    if not CONTEXT_PATH.is_dir():
        print(f'"{CONTEXT_PATH}" does not exist or is not a directory')
        raise RuntimeError()
    if not DOCKERFILE_PATH.is_file():
        print(f'"{DOCKERFILE_PATH}" does not exist or is not a file')
        raise RuntimeError()

    log_stream = docker.build(
        context_path=CONTEXT_PATH,
        # platforms=["linux/amd64"],
        platforms=["linux/arm64"],
        file=DOCKERFILE_PATH,
        tags="embed_text_service:latest_native_cpu",
        stream_logs=True,
    )
    log_stream = cast(Iterator[str], log_stream)
    for line in log_stream:
        print(line, end="")
