# Embed Text Container Service

If you're aiming to deploy a text embedding model to Snowpark Container Services, this code will help! This repo showcases a pure-Python approach to packaging a text embedding model into a Snowpark Container Services service.

## Prerequisites

You will need Python and Docker installed to use this code.

## Usage

**NOTE:** The best way to understand how to use this project is to check out the example walkthrough notebook `configurations/quickstart_example/walkthrough.ipynb`. Not only does this showcase every step that's required, but it also includes tricks (e.g. for debugging) and explanations of each step.

Usage involves implementing a custom *configuration* of this text embedding service inside the `configurations/` directory and then running your configuration. Copying the `configurations/quickstart_example` and editing it to suit your needs is a great way to implement and run a configuration of your own.

While the `quickstart_example` shows each step in a noteboo, your configuration need not utilize a Jupyter notebook at all. You can use whatever kind of scripting (e.g. Python scripting, shell scripting, Makefiles, etc.) that gets the job done, where the "job" in this context is to populate the `build` directory and then to use the `libs/buildlib` Python library to build and deploy a Docker image as a Snowpark Container Services Serivce.

This brings us to the definition of a *configuration*. A *configuration* is a script or set of scripts that does the following:

1. Add whatever data your custom service needs to bundle inside it into the `build/data` directory
   - Data includes model weights, custom Python libraries (packaged as wheel files or similar), etc.
2. Implement your embedding logic as an `embed.py` file that gets written to or copied into `/build/embed.py` -- see below for details
    - You must also define any dependencies for your `build/embed.py` in an accompanying `build/requirements.txt` file
3. Configure the service via creating a `build/config.py` file (see below for details)
4. Build your service's Docker image via `buildlib.build()`
    - Once you have built your image, you can optionally test it locally, e.g. via running it using `buildlib.start_locally()` (and `buildlib.stop_locally()` to stop it), then running the end-to-end test via `pytest testint/tests/test_end_to_end.py`
5. Push your service's Docker image to Snowflake via `buildlib.push()`
6. Deploy your service via `buildlib.deploy_service()`

### Implementing embedding

In this project, the core logic of text embedding is implemented as a single Python function called `get_embed_fn()`. This function does two things, load the model and any other data needed from the filesystem, and define an `embed` function that uses this (now loaded) model to transform a `Sequence[str]` of input into a matrix of embeddings (as a 2d `numpy` array of `np.float32` values).

For configuring your `get_embed_fn()` to load data (like model weights) correctly when it is packaged into the service Docker image, you may find it helpful to leverage the `BUILD_ROOT` environment variable, which will be preopoulated in the Docker image for you. The example below demonstrates one way you can use this environment variable when loading model weights.

``` python
# Example contents of `build/embed.py`
import os
from pathlib import Path
from typing import Sequence

import numpy as np

def get_embed_fn() -> Callable[[Sequence[str]], np.ndarray]:
    # Load the model into memory.
    data_dir = Path(os.environ["BUILD_ROOT"]) / "data"
    tokenizer = load_tokenizer(data_dir / "tokenizer")
    model = load_model(data_dir / "model")

    # Define an embedding function that maps a sequence of strings to a 2d embedding matrix.
    def embed(texts: Sequence[str]) -> np.ndarray:
        tokens = tokenizer.tokenize(texts)
        result_tensor = model.embed(tokens)
        result_array = result_tensor.cpu().numpy().astype(np.float32)
        assert result_array.ndim == 2
        assert result_array.shape[0] == len(texts)
        return result_array

    return embed
```

You will also need to define a `build/requirements.txt` file that instructs the build process to install any dependencies your `build/embed.py` code needs.

``` txt
# Example contents of `build/requirements.txt`
pytorch>=2.1
numpy>=0.24
transformers
```

### Configuring the service

The boring parts of the service (an API, a queue, and a process that pulls from the queue and calls your `embed` function) are already built for you and should work fine for a number of different applications with minimal alteration or configuration. However, some small amount of configuration is required, and a slightly greater amount is possible.

What is required? Just one thing: specifying the dimensionality of your embedding.

What else is configurable? Batch size, input queue length, and max input length.
- Batch size = maximum number of texts passed as input to your `embed` function
  - Bigger may be more efficient on GPU, but too big a size might incur memory issues
- Input queue length = maximum number of texts that the service will keep in memory at once
  - You can probably leave this as is in most cases, though you can increase from the default value if you are running a fast model or running on a machine type with many CPUs or a GPU
  - This value must be larger than the value of `external_function_batch_size` used in `buildlib.deploy_service`, or the service will refuse all full batches of input and cause queries to fail.
  - In general, the input queue length should allow the queue to store no more than 20-25 seconds of embedding work to avoid timeouts
  - If the warehouse tries to pass in more, the service will tell the warehouse to slow down
  - Making this too large can cause issues with canceled queries, since there is no mechanism to tell the service to remove items from the queue even if the query that sent in those items has been cancelled
  - Making this too large can cause issues with timeouts, since if it takes more than 30 seconds for an item to get through the queue, be embedded, and be returned, the warehouse will treat that as a timeout and try again, increasing load on the service
- Max input length = the longest string (in bytes) allowed as input
  - Passing in longer inputs will potentially trigger query failures
  - Large sizes increase memory usage, since memory is preallocated


``` python
# Example contents of `build/config.py
from service_config import Configuration

USER_CONFIG = Configuration(embedding_dim=768, max_batch_size=8)
```
  
### Building and deploying your custom service

It takes a number of steps to build and deploy a Docker image as a Snowpark Container Services service. To streamline the process, we offer a Python library called `buildlib` which offers a clear API to drive Docker and the Snowflake Python client into carrying out these steps for us.

To install `buildlib`, just use pip: `python -m pip install ./libs/buildlib`

To see `buildlib` in action, check out the example walkthrough notebook: `configurations/quickstart_example/walkthrough.ipynb`.

If you're curious, feel free to check out the source code, it's pretty short! Also, here's a quick synopsis of what `buildlib` does under the hood:
- `buildlib.build()` uses `python-on-whales` to run `docker build`
  - Calls the Docker CLI's `docker build` command with all the right arguments (standardized image name, explicit platform, the right path to the Dockerfile, etc.)
- `buildlib.push()` uses `python-on-whales` to run a few Docker commands
  - Optionally logs into your Snowflake Docker repository (`docker login`)
  - Re-tags your service Docker image to point to your Snowflake Docker repository (`docker tag`)
  - Pushes your service Docker image to Snowflake (`docker push`)
- `buildlib.deploy_service()` uses the Snowflake Python client to run all the Snowflake SQL you need to deploy the service
  - Templates out a service YAML spec and runs a `CREATE SERVICE ...` command using that spec
  - Runs several `CREATE FUNCTION ...` commands to set up an `EMBED_TEXT(VARCHAR)` SQL function in Snowflake that calls your new service


## Repo Overview

```
├── configurations             <- configurations you can adapt
│   ├── <examples>
│   ├── <put your code here>
├── dockerfiles
│   ├── Dockerfile             <- prebuilt recipe for the service Docker image
│   └── Dockerfile.dockerignore
├── libs
│   ├── buildlib               <- configures and executes Docker image building
│   ├── lodis                  <- provides queueing
│   ├── multiprocess_logging   <- provides logging
│   ├── service_config         <- defines the service's configuration convention
│   └── simple_lru_cache       <- provides caching for the API, currently not used
├── README.md
├── service_api                <- the API side of the service
├── service_embed_loop         <- the embedding side of the service
├── services_common_code       <- logic shared between the API and the embed loop
└── testing                    <- tools for testing your service locally
    ├── perf_test_scripts      <- longer running evaluations of performance
    └── tests                  <- fast running checks that things are working

```

## Contributing

If you'd like to contribute, please just open a pull request! We kindly ask that if you do open a PR, please take a minute to explain your changes in the PR description. Also please make sure you run the pre-commit hooks over you code and ensure tests as passing after your changes.

### Setting up your development environment

Everything here targets building a Linux Docker image, so you may run into issues if you are not developing on a Debian-like Linux environment. If you do run into issues, consider [developing inside a container with Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers), though be aware that this might make some of the Docker stuff trickier.

You may want to create a dedicated Python 3.8+ [virtual environment](https://docs.python.org/3/library/venv.html) for this part.

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

The libraries have a fair amount of test coverage. Use [pytest](https://pytest.org) to run the tests. Note that `lodis` and `multiprocess_logging` are currently Linux-only, so the tests will not pass on other OSes.

``` shell
pytest libs
```
