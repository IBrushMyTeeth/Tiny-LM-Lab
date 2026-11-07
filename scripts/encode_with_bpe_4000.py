"""
This module converts the raw training and validation datasets into flat
binary files containing token IDs. Each story is encoded independently
with BOS and EOS tokens to explicitly mark story boundaries.

To reduce memory usage, datasets are processed in batches rather than
loading all encoded stories into memory at once. Token IDs are flattened
into a continuous stream and stored as unsigned 16-bit integers
(uint16), which is sufficient for the configured 4,000-token vocabulary.

The resulting files can be efficiently loaded into NumPy arrays and
converted to PyTorch tensors for language model training.

Binary format:
- Each token is stored as a uint16 value (2 bytes).
- Stories are concatenated into a continuous token stream.
- No padding is added between stories.
- Story boundaries are represented by BOS and EOS tokens.
"""


from typing import cast

import numpy as np
from datasets import Dataset, load_from_disk

from project_paths import (
    RAW_TRAIN_DATASET,
    RAW_VALIDATION_DATASET,
    BPE_TOKENIZER_4000,
    PREPROCESSED_DATA,
)
from src.data_pipeline.tokenization import BPETokenizer


TRAIN_SAVE_PATH = PREPROCESSED_DATA / "train_bpe_4000.bin"
VALIDATION_SAVE_PATH = PREPROCESSED_DATA / "validation_bpe_4000.bin"

SPLITS = [
    (RAW_TRAIN_DATASET, TRAIN_SAVE_PATH),
    (RAW_VALIDATION_DATASET, VALIDATION_SAVE_PATH),
]

BATCH_SIZE = 5000

def main():
    tokenizer = BPETokenizer.load(BPE_TOKENIZER_4000)
    # If the vocabulary is larger than 65,535 then np.unit16 cannot be used
    assert tokenizer.vocab_size <= np.iinfo(np.uint16).max + 1

    for src, trg in SPLITS:
        ds = load_from_disk(src)
        assert isinstance(ds, Dataset)

        with open(trg, mode="wb") as f:
            for batch in ds.iter(batch_size=BATCH_SIZE):
                texts = cast(list[str], cast(dict, batch)["text"])
                encoded = tokenizer.encode_batch(
                    texts,
                    add_bos=True,
                    add_eos=True,
                )

                ids = (i for story in encoded for i in story)
                np.fromiter(ids, dtype=np.uint16).tofile(f)

        print(f"Tokenized: {src}")
        print(f"Saved to: {trg}")

if __name__ == "__main__":
    main()