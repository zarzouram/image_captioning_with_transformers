from collections import Counter
from typing import List


class DictWithDefault(dict):
    """Dictionary that return default value if key is missing.
    example:
        d = {"a": 1, "b": 2, "c": 3, "d": 4}
        dd = DictWithDefault("default", d)
        dd["e"] => "default"
        dd["a"] => 1
    """
    def __init__(self, default, *args):
        self.default = default
        super(DictWithDefault, self).__init__(*args)

    def __missing__(self, key):
        return self.default

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)


class Vocabulary:
    # map word to ints
    def __init__(self, words_list: List[str], min_freq: int = 3) -> None:

        word_count = Counter(words_list)
        word_filtered = [
            word for (word, cnt) in word_count.items() if cnt > min_freq
        ]
        word_filtered = ["<PAD>"] + word_filtered

        self.stoi = DictWithDefault(default=len(word_filtered))
        self.itos = DictWithDefault(default="<UNK>")

        for i, word in enumerate(word_filtered):
            self.stoi[word] = i
            self.itos[i] = word

    def __len__(self) -> int:
        # Get vocabulary size
        return len(self.stoi)
