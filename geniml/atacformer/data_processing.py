import os
from typing import List

from gtars.tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase


class TrainingTokenizer(Tokenizer, PreTrainedTokenizerBase):
    """
    A special training tokenizer. This class is a subclass of both **our** Tokenizer and
    PreTrainedTokenizerBase. This is because the data collator requires a collator that is
    both a Tokenizer and a PreTrainedTokenizerBase. This is a workaround to make the
    code work with our Tokenizer.
    """

    @property
    def added_tokens_decoder(self):
        return dict()

    @property
    def added_tokens_encoder(self):
        return dict()

    def num_special_tokens_to_add(self, pair=False):
        return len(self.special_tokens_map)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        # save vocab (a dict of token to id) to a file in the save_directory
        vocab = self.get_vocab()
        if not save_directory:
            raise ValueError("save_directory must be specified to save the vocabulary.")
        if filename_prefix:
            vocab_file = f"{filename_prefix}-vocab.txt"
        else:
            vocab_file = "vocab.txt"
        vocab_path = os.path.join(save_directory, vocab_file)
        with open(vocab_path, "w", encoding="utf-8") as vocab_writer:
            for token, token_id in vocab.items():
                vocab_writer.write(f"{token}\t{token_id}\n")

        return (vocab_path,)

    @property
    def all_special_ids(self):
        """
        Returns a list of all special token ids.
        """
        return self.encode(list(self.special_tokens_map.values()))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        PreTrainedTokenizerBase.__init__(self)
