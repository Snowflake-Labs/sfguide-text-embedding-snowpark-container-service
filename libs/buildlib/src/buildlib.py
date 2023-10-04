from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import cast
from typing import Iterator
from typing import Optional

from python_on_whales import Container
from python_on_whales import docker

SERVICE_NAME = "embed_text_service"


def build(build_dir: Path, platform: Optional[str] = None, tag: str = "latest") -> None:
    # Validate the `build_dir` is correct, that we have a Dockerfile, etc.
    if not build_dir.is_dir():
        raise ValueError()
    if not build_dir.name == "build":
        raise ValueError()
    context_path = build_dir.resolve().parent
    dockerfile_path = context_path / "dockerfiles" / "Dockerfile"
    if not dockerfile_path.is_file():
        print(f'"{dockerfile_path}" does not exist or is not a file')
        raise RuntimeError()

    # Run the build.
    log_stream = docker.build(
        context_path=context_path,
        platforms=[platform] if platform is not None else None,
        file=dockerfile_path,
        tags=f"{SERVICE_NAME}:{tag}",
        stream_logs=True,
    )
    log_stream = cast(Iterator[str], log_stream)
    for line in log_stream:
        print(line, end="")


def _is_healthy(container: Container) -> bool:
    health = container.state.health
    if health is None:
        return False
    status = health.status
    if status is None:
        return False
    return status.lower() == "healthy"


def stop_locally() -> None:
    docker.remove(containers=SERVICE_NAME, force=True)


def start_locally(tag: str = "latest") -> None:
    stop_locally()
    container = docker.run(
        image=f"{SERVICE_NAME}:{tag}", networks=["host"], name=SERVICE_NAME, detach=True
    )
    container = cast(Container, container)

    # Wait a bit for the container to come up.
    for _ in range(30 * 20):
        if _is_healthy(container):
            break
        sleep(1 / 20)
    if not _is_healthy(container):
        print("Waited about 30 seconds, but the container is still not healthy!")


@contextmanager
def run_container_context(tag: str = "latest"):
    try:
        start_locally(tag)
        yield
    finally:
        stop_locally()
