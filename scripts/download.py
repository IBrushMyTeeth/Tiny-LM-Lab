"""
Download and prepare local subsets of the TinyStories dataset.

Run this script before training to download the Hugging Face TinyStories
dataset and save subsets of its training and validation splits to the
project's data directory.

The script selects 120,000 training examples and 6,000 validation examples
using a fixed random seed. The resulting subsets are saved locally and can
be reused during training without downloading the dataset again.

The full dataset is downloaded and prepared in a temporary cache directory
before sampling. This approach allows shuffling across the entire dataset,
avoiding the sampling limitations of a finite streaming buffer and the need
for manual shard-level sampling.

Once all requested subsets have been saved successfully, the cache is
deleted to free disk space.

Dataset source: https://huggingface.co/datasets/roneneldan/TinyStories
"""


import shutil

from project_paths import DATA_DIR
from datasets import load_dataset


CACHE_DIR = DATA_DIR / "cache"
SEED = 42

SPLITS = {
    "train": 120_000,
    "validation": 6_000,
}


def main():
    missing = {s: n for s, n in SPLITS.items() if not (DATA_DIR / s).exists()}

    if missing:
        # A DatasetDict with all splits is downloaded in a temporary cache
        full = load_dataset(
            "roneneldan/TinyStories",
            cache_dir=str(CACHE_DIR),
        )

        for split, num_samples in missing.items():
            ds = full[split]
            subset = ds.shuffle(seed=SEED).select(range(min(num_samples, len(ds))))
            subset.save_to_disk(str(DATA_DIR / split))
            print(f"Saved {len(subset)} samples to {DATA_DIR / split}")

    # If splits were created successfully, delete the cache 
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Deleted full dataset cache at {CACHE_DIR}")


if __name__ == "__main__":
    main()