from typing import DefaultDict, List, Dict, Tuple
from collections import defaultdict, Counter
import re

captions = DefaultDict[str, List[List[str]]]
images_w_captions = Dict[str, captions]


class Vocabulary:
    # map word to ints
    def __init__(self, words_list: List[str], min_freq: int = 3) -> None:

        word_count = Counter(words_list)
        word_filtered = [
            word for (word, cnt) in word_count.items() if cnt > min_freq
        ]

        self.stoi = {}
        self.itos = {}
        word_filtered = ["<PAD>"] + word_filtered + ["<UNK>"]

        for i, word in enumerate(word_filtered):
            self.stoi[word] = i
            self.itos[i] = word

    def __len__(self) -> int:
        # Get vocabulary size
        return len(self.stoi)


def get_captions(annotations: list) -> Tuple[captions, List[str]]:
    """ Images and thier captions are separated into two list of dicts.

    json_path: a string of the mscoco annotation file
    """

    # collect captions by image id
    captions_dict = defaultdict(list)
    all_captions = []
    for annton in annotations:
        captions = [
            s for s in re.split(r"(\W)", annton["caption"]) if s.strip()
        ]
        captions = ["<SOS>"] + captions + ["<EOS>"]
        all_captions.extend(captions)
        captions_dict[annton["image_id"]].append(captions)

    return captions_dict, all_captions


def combine_image_captions(images: List[str],
                           captions_dict: captions) -> images_w_captions:
    """ Images and thier captions are separated into two list of dicts.

    json_path: a string of the mscoco annotation file
    """

    # collect image and captions
    images_w_captions = {}
    for img in images:
        img_id = img["id"]

        images_w_captions[img["file_name"]] = {
            "image_id": img_id,
            "captions": captions_dict[img_id]
        }

    return images_w_captions
