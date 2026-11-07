"""
Provide a reusable wrapper around a ByteLevel BPE tokenizer.

The BPETokenizer class provides a consistent interface for training,
saving, loading, encoding, and decoding text. It also manages the PAD,
BOS, and EOS special tokens used by the language model.

A trained tokenizer can be saved and reused across dataset preprocessing,
model training, and text generation.
"""


from pathlib import Path
from collections.abc import Iterable

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


PAD, BOS, EOS = "<|PAD|>", "<|BOS|>", "<|EOS|>"
SPECIAL_TOKENS = ["<|PAD|>", "<|BOS|>", "<|EOS|>"]

class BPETokenizer:
    """Wrap a BPE tokenizer with special-token handling and convenience methods."""
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        if tokenizer is None:
            tokenizer = Tokenizer(BPE())
            # Domain-specific choice: stories are independent and do not start
            # with whitespace, so add a prefix space for consistent encoding.
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            tokenizer.decoder = decoders.ByteLevel()

        self.tokenizer = tokenizer

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )

        self.tokenizer.train_from_iterator(
            iterator=texts,
            trainer=trainer,
        )
        self._check_special_tokens()

    def save(self, path: Path) -> None:
        self.tokenizer.save(path=str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        if not path.is_file():
            raise FileNotFoundError(path)
        
        tokenizer = Tokenizer.from_file(str(path))
        instance = cls(tokenizer)
        instance._check_special_tokens()
        return instance

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        id = self.tokenizer.token_to_id(PAD)
        if id is None:
            raise ValueError(f"Tokenizer doesnt recognize: {PAD}")
        return id

    @property
    def bos_token_id(self) -> int:
        id = self.tokenizer.token_to_id(BOS)
        if id is None:
            raise ValueError(f"Tokenizer doesnt recognize: {BOS}")
        return id

    @property
    def eos_token_id(self) -> int:
        id = self.tokenizer.token_to_id(EOS)
        if id is None:
            raise ValueError(f"Tokenizer doesnt recognize: {EOS}")
        return id

    def _check_special_tokens(self) -> None:
        for token in SPECIAL_TOKENS:
            if self.tokenizer.token_to_id(token) is None:
                raise RuntimeError(f"Missing special token: {token}")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        tokens += self.tokenizer.encode(text).ids

        if add_eos:
            tokens.append(self.eos_token_id)

        return tokens

    def encode_batch(
        self, texts: list[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        # Parallelized in Rust, much faster than a Python loop over encode()
        encodings = self.tokenizer.encode_batch(texts)
        out = []
        for e in encodings:
            ids = e.ids
            if add_bos:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]
            out.append(ids)
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        # Undo the prefix space from add_prefix_space=True
        return text[1:] if text.startswith(" ") else text