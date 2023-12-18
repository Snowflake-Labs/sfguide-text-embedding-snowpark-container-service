from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    """
    - `embedding_dim`: The output dimensionality of your model.
    - `max_batch_size`: The maximum batch size to use when performing embedding.
        On CPU, this doesn't need to be big for efficienc hardware usage, even
        for small models.
    - `max_input_queue_length`: The maximum number of inputs to internally queue.
        This should stay short to mitigate the wasted work of cancelled queries.
    - `max_input_bytes`: Since the input queue uses preallocated memory, we need
        to determine how much memory to preallocate per item. 10k bytes per input
        is equivalent to around 2k words of average English text, and perhaps a
        bit more than 2k BERT tokens.

    NOTE: Docker defaults to setting `/dev/shm` (the in-memory filesystem where
    the queue lives) to only 64MiB, so the total size of the input queue
    (max input length * max_input_bytes) should stay pretty small.
    Consider leaving `max_input_queue_length` and `max_input_bytes` at default.
    """

    embedding_dim: int
    max_batch_size: int = 4
    max_input_queue_length: int = 16
    max_input_bytes: int = 10_000
