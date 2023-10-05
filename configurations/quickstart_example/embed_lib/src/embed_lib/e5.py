import logging
from dataclasses import dataclass
from typing import cast
from typing import List
from typing import Mapping
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from ._batch_iter_util import iter_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class E5Model:
    tokenizer: BertTokenizerFast
    model: BertModel


def load_e5_model(size: str = "large") -> E5Model:
    assert size in ("small", "base", "large")
    model_name = f"intfloat/e5-{size}-v2"
    logging.info(f"Loading model and tokenizer from Huggingface: `{model_name}`")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = cast(BertModel, BertModel.from_pretrained(model_name))
    return E5Model(tokenizer, model)


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _tokenize_batch(
    tokenizer: BertTokenizerFast, texts: List[str]
) -> Mapping[str, Tensor]:
    return tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )


def _embed_batch(
    model: BertModel, batch_dict: Mapping[str, Tensor], normalize: bool
) -> Tensor:
    outputs = model(**batch_dict)
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def _tokenize_and_embed_batch(
    e5_model: E5Model, texts: List[str], normalize: bool
) -> Tensor:
    batch_dict = _tokenize_batch(e5_model.tokenizer, texts)
    embeddings = _embed_batch(e5_model.model, batch_dict, normalize=normalize)
    return embeddings


def embed(
    e5_model: E5Model,
    texts: Sequence[str],
    batch_size: int = 8,
    normalize: bool = True,
    progress_bar: bool = True,
) -> Tensor:
    res_batches = []
    pbar = tqdm(total=len(texts), disable=not progress_bar, desc="embedding")
    for chunk in iter_chunks(texts, batch_size):
        with torch.no_grad():
            res = _tokenize_and_embed_batch(e5_model, chunk, normalize=normalize)
        pbar.update(len(chunk))
        res_batches.append(res)

    return torch.cat(res_batches)
