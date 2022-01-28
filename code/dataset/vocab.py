from typing import Dict, Optional
import json
from .utils import write_json


class DictWithDefault(dict):
    """Dictionary that return default value if key is missing.
    example:
        d = {"a": 1, "b": 2, "c": 3, "d": 4}
        dd = DictWithDefault("default", d)
        dd["e"] => "default"
        dd["a"] => 1
    """

    def __init__(self, default, *args):
        self.default = default  # set a default value to return
        super(DictWithDefault, self).__init__(*args)

    def __missing__(self, key):
        # key not in the dict return the default value
        return self.default

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)


class Vocabulary:
    """map word to ints, and ints to word. If word does not exist return
    the index of the <UNK> "unkown token"
    """

    def __init__(self,
                 words_counter: Optional[Dict[str, int]] = None,
                 min_freq: int = 5) -> None:

        self.words_counter = words_counter
        self.min_freq = min_freq
        if words_counter:
            self.build_vocab()

    def load_vocab(self, words_list_path: str, min_freq: Optional[int] = None):
        with open(words_list_path) as json_file:
            data = json.load(json_file)

        self.words_counter = data["words_counter"]
        self.min_freq = min_freq if min_freq is not None else data["min_freq"]
        self.build_vocab()

    def save_vocab(self, save_path):
        data = {"words_counter": self.words_counter, "min_freq": self.min_freq}
        write_json(save_path, data)

    def build_vocab(self):
        # remove the words that are repeated less than or equal to min_fre
        word_filtered = [
            word for (word, cnt) in self.words_counter.items()
            if cnt > self.min_freq
        ]
        word_filtered = ["<PAD>"] + word_filtered  # add PAD token, index=0

        # Setting "string to index" stoi and "index to string" itos
        # dectionaries
        self.stoi = DictWithDefault(default=len(word_filtered))
        self.itos = DictWithDefault(default="<UNK>")

        for i, word in enumerate(word_filtered):
            self.stoi[word] = i
            self.itos[i] = word

    def __len__(self) -> int:
        # Get vocabulary size
        return len(self.stoi) + 1
