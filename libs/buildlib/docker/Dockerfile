FROM python:3.8.16-bullseye
WORKDIR /root
ENV BUILD_ROOT="/root"

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
    # Overwrite the placeholder embedding logic with the user-specified logic.
    && mv ./build/embed.py ./service_embed_loop/embed.py
RUN ./venv_embed_loop/bin/python -m pip install -r ./build/requirements.txt

COPY ./build/embed.py ./service_embed_loop/embed.py

# Run the API application.
ENV PYTHONPATH=.
CMD ["./venv_api/bin/python", "./service_api/main.py"]
