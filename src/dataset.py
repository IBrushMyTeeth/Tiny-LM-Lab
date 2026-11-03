"""
PyTorch Dataset wrapper for loading pre-tokenized TinyStories data.

The dataset reads a flat binary file containing token IDs stored as
unsigned 16-bit integers (uint16). The entire file is loaded into RAM
using NumPy's fromfile function. Mainly to avoid repeated disk access.

Stories are stored as a continuous token stream, with BOS and EOS tokens
marking individual story boundaries. The dataset does not treat these
boundaries specially. Consequently, training sequences may span multiple
stories, allowing the model to learn from the continuous token stream.

Both input and target sequences are returned as NumPy arrays with dtype
uint16. NumPy slicing returns views into the underlying array rather
than allocating copies of the token data.

PyTorch's default DataLoader collation converts the returned NumPy arrays
into batched tensors. These tensors should subsequently be converted
to torch.long during batch processing, as required by embedding layers
and loss calculations.
"""


from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    def __init__(
            self, path: Path,
            context_length: int,
            stride: int,
        ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")

        # self.data = np.memmap(path, dtype=np.uint16, mode="r")

        # Load the entire tokenized dataset into RAM.
        # This avoids disk access during training and is feasible given
        # the relatively small size of the dataset.
        # Use np.memmap instead if memory becomes a constraint.
        self.data = np.fromfile(path, dtype=np.uint16)
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        return (len(self.data) - self.context_length - 1) // self.stride + 1

    def __getitem__(
        self, idx: int
    ) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Dataset index out of range.")
        
        start = idx * self.stride
        end = start + self.context_length

        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        return x, y