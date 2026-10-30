"""
Train and save a BPE tokenizer.

Loads the locally stored TinyStories training dataset, trains a
BPETokenizer with a vocabulary size of 4,000, and saves the
resulting tokenizer configuration.

Only the training split is used to avoid data leakage from the
validation set.
"""


from project_paths import RAW_TRAIN_DATASET, TOKENIZER_DIR

from datasets import Dataset, load_from_disk
from src.tokenization import BPETokenizer


SAVE_PATH = TOKENIZER_DIR / "bpe_4000.json"

def main():
    ds = load_from_disk(RAW_TRAIN_DATASET)
    assert isinstance(ds, Dataset)

    tokenizer = BPETokenizer()
    tokenizer.train(
        texts=ds["text"],
        vocab_size=4000,
    )

    tokenizer.save(SAVE_PATH)

if __name__ == "__main__":
    main()