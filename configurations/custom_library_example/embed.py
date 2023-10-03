import logging
from typing import Callable
from typing import cast
from typing import Sequence

import embed_lib.e5
import numpy as np
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

MAX_BATCH_SIZE = 4


def get_embed_fn(logger: logging.Logger) -> Callable[[Sequence[str]], np.ndarray]:
    # Load the model into memory.
    logger.info("[_get_embed_fn]Loading model from disk to memory")
    tokenizer = BertTokenizerFast.from_pretrained(
        "/root/data/tokenizer", local_files_only=True
    )
    model = cast(
        BertModel, BertModel.from_pretrained("/root/data/model", local_files_only=True)
    )
    e5_model = embed_lib.e5.E5Model(tokenizer, model)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        result_tensor = embed_lib.e5.embed(
            e5_model=e5_model,
            texts=texts,
            batch_size=MAX_BATCH_SIZE,
            normalize=True,
            progress_bar=False,
        )
        result_array = result_tensor.numpy().astype(np.float32)
        return result_array

    return _embed
