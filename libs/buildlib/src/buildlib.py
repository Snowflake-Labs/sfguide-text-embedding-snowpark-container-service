from pathlib import Path
from typing import cast
from typing import Iterator

from python_on_whales import docker


def build(build_dir: Path) -> None:
    if not build_dir.is_dir():
        raise ValueError()
    if not build_dir.name == "build":
        raise ValueError()
    context_path = build_dir.resolve().parent
    dockerfile_path = context_path / "dockerfiles" / "Dockerfile"
    if not dockerfile_path.is_file():
        print(f'"{dockerfile_path}" does not exist or is not a file')
        raise RuntimeError()
    log_stream = docker.build(
        context_path=context_path,
        # TODO: Switch to amd64, switch tag, etc. to support more flexible builds.
        # platforms=["linux/amd64"],
        platforms=["linux/arm64"],
        file=dockerfile_path,
        tags="embed_text_service:latest_native_cpu",
        stream_logs=True,
    )
    log_stream = cast(Iterator[str], log_stream)
    for line in log_stream:
        print(line, end="")
