import shutil
from pathlib import Path

import buildlib
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from build import ProjectBuilder

BUILD_DIR = Path(__file__).resolve().parents[2] / "build"


def main() -> None:
    prepare_build_directory()
    buildlib.build()


def prepare_build_directory() -> None:
    """Prepare the `build` directory to contain code and data."""
    shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()
    copy_requirements_txt_and_embed_py()
    build_custom_library()
    download_model_weights()


def download_model_weights() -> None:
    MODEL_NAME = "intfloat/e5-base-v2"
    SAVE_DIR = BUILD_DIR / "data"
    TOKENISER_DIR = SAVE_DIR / "tokenizer"
    MODEL_DIR = SAVE_DIR / "model"

    print(f"Downloading {MODEL_NAME} and saving tokenizer and weights to {SAVE_DIR}")

    # Download the tokenizer and model and save copies to specific local directories.
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    assert isinstance(model, BertModel)  # For typechecking.
    tokenizer.save_pretrained(TOKENISER_DIR)
    model.save_pretrained(MODEL_DIR)

    # Validate that our saved files work by loading from them.
    tokenizer = BertTokenizerFast.from_pretrained(TOKENISER_DIR)
    model = BertModel.from_pretrained(MODEL_DIR)


def build_custom_library() -> None:
    wheel_filename = ProjectBuilder(Path(__file__).parent / "libs" / "embed_lib").build(
        distribution="wheel", output_directory=BUILD_DIR / "data"
    )
    print(f"Built {wheel_filename}")


def copy_requirements_txt_and_embed_py() -> None:
    PARENT_DIR = Path(__file__).parent
    shutil.copy2(PARENT_DIR / "requirements.txt", BUILD_DIR / "requirements.txt")
    shutil.copy2(PARENT_DIR / "embed.py", BUILD_DIR / "embed.py")


if __name__ == "__main__":
    main()
