# Embed Text Container Service

If you're aiming to deploy a text embedding model to Snowpark Container Services, this code will help! This repo showcases a pure-Python approach to packaging a text embedding model into a Snowpark Container Services service.

## Usage

Basic usage involves checking out the examples in the `configurations` directory, picking the example most similar to your needs, copying it, and adapting it.


### Overview of a configuration

```
TODO: Explain what a configuration should look like!
```

## Repo Overview

```
├── configurations             <- configurations you can adapt
│   ├── <examples>
│   ├── <put your code here>
├── libs
│   ├── buildlib               <- builds your service's Docker image
│   ├── lodis                  <- provides queueing
│   ├── multiprocess_logging   <- provides logging
│   └── simple_lru_cache       <- provides caching
├── README.md
├── service_api                <- the API side of the service
├── service_embed_loop         <- the embedding side of the service
├── services_common_code       <- logic shared between the API and the embed loop
└── testing                    <- tools for testing your service locally
    ├── perf_test_scripts          <- longer running evaluations of performance
    └── tests                      <- fast running checks that things are working

```

### For users

Unless you need to do heavy customization, you can stick primarily inside the `configurations` directory. Once you've taken a stab at creating your own configuration, you'll also probably want to use the tests inside the `testing` directory to make sure things are working as expected.

If you're curious how things come together, e.g. you want to know how the `libs/buildlib` library works to build your Docker image, take a look at the source code -- it's actually fairly straightforward stuff! And of course, nothing stops you from making changes to improve your text embedding service for your needs. Read on

### For developers

```
TODO: Explain
```


## Contributing

If you'd like to contribute, please just open a pull request! We kindly ask that if you do open a PR, please take a minute to explain your changes in the PR description. Also please make sure you run the pre-commit hooks over you code and ensure tests as passing after your changes.

### Setting up your development environment

Everything here targets building a Docker image based off of the `python:3.8.16-bullseye` base image, so you may run into issues if you are not developing on a Debian-like Linux environment and using Python 3.8. If you do run into issues, consider [developing inside a container with Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers).

You may want to create a dedicated Python 3.8 [virtual environment](https://docs.python.org/3/library/venv.html) for this part.

``` shell
# If you want nice IDE autocomplete, the dependencies for all the services.
# If running from a context besides the root of this repo, edit
# `BUILD_ROOT=$(pwd)` to express the correct path to the root of this repo.
BUILD_ROOT=$(pwd) pip install ${BUILD_ROOT}/service_api/requirements.txt \ 
    && pip install -r ${BUILD_ROOT}/service_embed_loop/requirements.txt \ 
    && pip install -r ${BUILD_ROOT}/services_common_code/requirements.txt

# Install the packages in `libs` in editable mode.
ls --directory -1 libs/* | xargs -I {} pip install -e {}

# Install `pytest` to run the tests.
pip install pytest

# Set up precommit.
pre-commit install
```

### Running tests

The libraries have a fair amount of test coverage. Use [pytest](https://pytest.org) to run the tests.

``` shell
pytest libs
```
