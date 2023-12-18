from contextlib import contextmanager
from getpass import getpass
from io import StringIO
from pathlib import Path
from textwrap import dedent
from time import sleep
from typing import cast
from typing import Iterator
from typing import Literal
from typing import Optional

import snowflake.connector.util_text
from python_on_whales import Container
from python_on_whales import docker

SERVICE_NAME = "embed_text_service"
SPEC_FILE = "embed_text_service.yaml"


def build(
    build_dir: Path,
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64",
    tag: str = "latest",
) -> None:
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


def push(
    repo_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    tag: str = "latest",
    skip_login: bool = False,
) -> None:
    # Log in.
    if not skip_login:
        if username is None:
            username = input("Username: ")
        if password is None:
            password = getpass(f"Password for {username}: ")
        docker.login(repo_url, username=username, password=password)

    # Retag from docker.io/embed_text_service:latest to
    # ....registry.snowflakecomputing.com/..../embed_text_service:latest .
    default_repo_name = f"{SERVICE_NAME}:{tag}"
    specified_repo_name = f"{repo_url}/{default_repo_name}"
    docker.tag(default_repo_name, specified_repo_name)

    # Push!
    print(f"Pushing {specified_repo_name}")
    docker.push(specified_repo_name)


def _run_sql(
    connection: snowflake.connector.connection.SnowflakeConnection, command: str
) -> None:
    statements = snowflake.connector.util_text.split_statements(StringIO(command))
    for statement, _is_put_or_get in statements:
        print(statement)
        connection.execute_string(statement)


def deploy_service(
    connection: snowflake.connector.connection.SnowflakeConnection,
    embedding_dim: int,
    role: str,
    database: str,
    schema: str,
    spec_stage: str,
    compute_pool: str,
    image_repository: str,
    image_database: Optional[str] = None,
    image_schema: Optional[str] = None,
    image_tag: str = "latest",
    min_instances: int = 1,
    max_instances: int = 1,
    external_function_batch_size: int = 4,
) -> None:
    """
    Deployment prerequisites:
    - a database
    - a schema
    - an image repository
    - a role that can create stages and services in the db/schema using images
      from the image repository

    Deployment entails the following:
    - USE-ing the role, database, and schema
    - Ensuring the service spec stage is created (if it doesn't already exist)
    - Creating and uploading a service spec yaml file
    - Creating the service
    - Creating SQL functions that enable usage of the service
    """
    if image_repository.startswith("@"):
        image_repository = image_repository[1:]
    if spec_stage.startswith("@"):
        spec_stage = spec_stage[1:]
    if image_database is None:
        image_database = database
    if image_schema is None:
        image_schema = schema

    # Use the right role, database, and schema.
    _run_sql(
        connection, f"use role {role}; use database {database};" f"use schema {schema};"
    )

    # Create the service spec.
    spec_yaml = dedent(
        f"""
        spec:
          containers:
            - name: {SERVICE_NAME.replace("_", "-")}
              image: /{image_database}/{image_schema}/{image_repository}/{SERVICE_NAME}:{image_tag}
              readinessProbe:
                port: 8000
                path: /healthcheck
          endpoint:
            - name: endpoint
              port: 8000
        """
    )

    # Create the service.
    _run_sql(connection, f"drop service if exists {SERVICE_NAME};")
    create_statement = dedent(
        f"""
        create service {SERVICE_NAME}
            in compute pool {compute_pool}
            from specification\n$${spec_yaml}$$
            min_instances = {min_instances}
            max_instances = {max_instances};
        """
    )
    _run_sql(connection, create_statement)

    # Create the SQL functions needed to use the service.
    embed_to_base64_create_statement = dedent(
        f"""
        create or replace function _embed_to_base64(input string)
            returns string
            service={SERVICE_NAME}!endpoint
            max_batch_rows={external_function_batch_size}
            as '/embed';
        """
    )
    unpack_binary_array_create_statement = dedent(
        """
        create or replace function _unpack_binary_array(B binary)
            returns array
            language javascript
            immutable
            as
            $$
                return Array.from(new Float32Array(B.buffer));
            $$;
        """
    )
    embed_text_create_statement = dedent(
        f"""
        create or replace function embed_text(input string)
            returns vector(float,{embedding_dim})
            language SQL
            as
            $$_unpack_binary_array(to_binary(_embed_to_base64(input), 'BASE64'))::vector(float,{embedding_dim})$$;
        """
    )
    _run_sql(connection, embed_to_base64_create_statement)
    _run_sql(connection, unpack_binary_array_create_statement)
    _run_sql(connection, embed_text_create_statement)


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
        image=f"{SERVICE_NAME}:{tag}",
        name=SERVICE_NAME,
        detach=True,
        publish=[("8000", "8000")],
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
