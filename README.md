# Embed Text Container Service

This repo showcases a pure-Python implementation of deploying a text embedding model to Snowpark Container Services.

## Usage

```
TODO: Explain the process of using this code.
```


## Repo Overview

```
# TODO: Add annotated output of `tree` here!
```

## Contributing

- TOOD: describe setting up IDE
- TODO: describe installing local packages
- TODO: describe precommit setup

``` shell
# Install all the dependencies.
EMBED_TEXT_ROOT=$(pwd) pip install -r requirements_dev.txt

# Reinstall the stuff in `libs` in editable mode.
ls --directory -1 libs/* | xargs -I {} pip install -e {}

# Set up precommit.
pre-commit install
```

Testing

``` shell
pytest libs
```