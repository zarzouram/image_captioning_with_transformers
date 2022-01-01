from typing import List, DefaultDict, Dict, Tuple
from argparse import Namespace

from pathlib import Path
from itertools import chain

from sklearn.model_selection import train_test_split

from dataset.vocab import Vocabulary
from dataset.utils import get_captions, combine_image_captions, load_json
from dataset.utils import parse_arguments

captions_t = DefaultDict[str, List[List[str]]]
images_w_captions_t = Dict[str, captions_t]


def get_data(json_path: str) -> images_w_captions_t:
    """Load annations json file and return a images ids with its captions in
        the following format:
            image_name: {image_id: list of captions tokens}
    """

    annotations, images_id = load_json(json_path)
    captions = get_captions(annotations)
    images_w_captions = combine_image_captions(images_id, captions)

    return images_w_captions


def split_dataset(
    original_train_split: images_w_captions_t,
    original_val_split: images_w_captions_t,
    test_perc: int = 0.15,
    val_perc: int = 0.15
) -> Tuple[images_w_captions_t, images_w_captions_t, images_w_captions_t]:
    """The size of the original validation split is 4% of the dataset. The
        function calculate the remaining percentage to have a test set of size
        15% of the dataset. Then split the remaining to have a validation
        dataset of 15%.
    """
    train_perc = 1 - (test_perc + val_perc)
    original_val_size = len(original_val_split)
    original_train_size = len(original_train_split)
    ds_size = original_val_size + original_train_size

    # Calculate the remaining size to have 15% of test split (original_val +
    # test_makup)
    test_makup_size = int(ds_size * val_perc) - original_val_size
    train_size = int((train_perc / (1 - test_perc)) *
                     (ds_size - original_val_size - test_makup_size))

    original_train_list = list(original_train_split.items())
    test_makup, train_val = train_test_split(original_train_list,
                                             train_size=test_makup_size,
                                             random_state=42,
                                             shuffle=True)

    test_split = {**dict(test_makup), **original_val_split}  # merge two dicts

    train_split, val_split = train_test_split(train_val,
                                              train_size=train_size,
                                              random_state=42,
                                              shuffle=True)

    return dict(train_split), dict(val_split), test_split


def buil_vocab(captions: List[chain]) -> Vocabulary:
    all_words = list(chain.from_iterable(captions))
    return Vocabulary(all_words)


if __name__ == "__main__":

    # parse argument command
    args = parse_arguments()  # type: Namespace

    ds_dir = Path(args.dataset_dir)  # dataset directory

    file_path = str(ds_dir / args.json_train)  # train ann path
    images_captions = get_data(file_path)  # process train anntn file

    file_path = str(ds_dir / args.json_val)
    images_captions_test = get_data(file_path)  # process val anntn file

    # split dataset
    train_ds, val_ds, test_ds = split_dataset(images_captions,
                                              images_captions_test)
    # Create vocab from train dataset srt OOV to <UNK>
    captions = [chain.from_iterable(d["captions"]) for d in train_ds.values()]
    vocab = buil_vocab(captions)
