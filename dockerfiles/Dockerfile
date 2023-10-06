FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ARG TARGETPLATFORM
ARG PYTHON_VERSION=3.8
WORKDIR /root
ENV BUILD_ROOT="/root"

# Install Python via miniconda.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl && \
    rm -rf /var/lib/apt/lists/*
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
         *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/miniconda.sh -O  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install the core application (not user code).
COPY ./libs ./libs
COPY ./service_api ./service_api
COPY ./service_embed_loop ./service_embed_loop
COPY ./services_common_code ./services_common_code

RUN python -m venv --copies --prompt api create venv_api \
    && python -m venv --copies --prompt embed_loop create venv_embed_loop \
    && ./venv_api/bin/python -m pip install -r ./services_common_code/requirements.txt \
    && ./venv_api/bin/python -m pip install -r ./service_api/requirements.txt \
    && ./venv_embed_loop/bin/python -m pip install -r ./services_common_code/requirements.txt \
    && ./venv_embed_loop/bin/python -m pip install -r ./service_embed_loop/requirements.txt

# Install the user model and code.
COPY ./build ./build
RUN \
    # Bring in artifacts from the user.
    mv ./build/data ./data \
    # Overwrite the placeholder config with the user-specified config.
    && mv ./build/config.py ./services_common_code/config.py \
    # Overwrite the placeholder embedding logic with the user-specified logic.
    && mv ./build/embed.py ./service_embed_loop/embed.py
RUN ./venv_embed_loop/bin/python -m pip install -r ./build/requirements.txt

COPY ./build/embed.py ./service_embed_loop/embed.py

# Run the API application.
ENV PYTHONPATH=.
HEALTHCHECK --interval=5s --timeout=1s CMD curl --fail http://localhost:8000/healthcheck || exit 1
CMD ["./venv_api/bin/python", "./service_api/main.py"]
