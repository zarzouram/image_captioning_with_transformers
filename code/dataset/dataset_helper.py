from typing import List, Tuple
from numpy.typing import NDArray
from .custom_types import Captions, ImagesAndCaptions

from collections import defaultdict
from itertools import chain
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import re

import numpy as np
from sklearn.model_selection import train_test_split

import cv2

from .vocab import Vocabulary


def get_captions(annotations: list) -> Captions:
    """ Images and thier captions are separated into two list of dicts.

    json_path: a string of the mscoco annotation file
    """

    # collect captions by image id
    captions_dict = defaultdict(list)
    for annton in annotations:
        captions = [
            s for s in re.split(r"(\W)", annton["caption"]) if s.strip()
        ]
        captions = ["<SOS>"] + captions + ["<EOS>"]
        captions_dict[annton["image_id"]].append(captions)

    return captions_dict


def combine_image_captions(images: List[str], captions_dict: Captions,
                           images_dir: str) -> ImagesAndCaptions:
    """ Images and thier captions are separated into two list of dicts.

    json_path: a string of the mscoco annotation file
    """

    # collect image and captions
    images_w_captions = {}
    for img in images:
        img_id = img["id"]

        img_filename = images_dir + "/" + img["file_name"]
        images_w_captions[img_filename] = {
            "image_id": img_id,
            "captions": captions_dict[img_id]
        }

    return images_w_captions


def load_images(image_path: str,
                resize_h: int = None,
                resize_w: int = None) -> NDArray:
    img = cv2.resize(cv2.imread(image_path), (resize_h, resize_w),
                     interpolation=cv2.INTER_AREA)  # type: NDArray
    return img.transpose(2, 0, 1)


def encode_captions(captions: List[List[str]],
                    vocab: Vocabulary) -> Tuple[List[List[int]], List[int]]:
    """Encode captions text to the respective indices"""
    encoded = []
    lengthes = []
    for caption in captions:
        encoded.append([vocab.stoi[s] for s in caption])
        lengthes.append(len(caption))

    return encoded, lengthes


def split_dataset(
    original_train_split: ImagesAndCaptions,
    original_val_split: ImagesAndCaptions,
    SEED: int,
    test_perc: int = 0.15,
    val_perc: int = 0.15
) -> Tuple[ImagesAndCaptions, ImagesAndCaptions, ImagesAndCaptions]:
    """The size of the original validation split is 4% of the dataset. The
        function calculate the remaining percentage to have a test set of size
        15% of the dataset. Then split the remaining to have a validation
        dataset of 15%.
    """
    train_perc = 1 - (test_perc + val_perc)  # training %
    original_val_size = len(original_val_split)
    original_train_size = len(original_train_split)
    ds_size = original_val_size + original_train_size  # dataset size

    # Calculate the remaining size to have 15% of test split (original_val +
    # test_makup)
    test_makup_size = int(ds_size * val_perc) - original_val_size
    train_size = int((train_perc / (1 - test_perc)) *
                     (ds_size - original_val_size - test_makup_size))

    original_train_list = list(original_train_split.items())
    test_makup, train_val = train_test_split(original_train_list,
                                             train_size=test_makup_size,
                                             random_state=SEED,
                                             shuffle=True)

    test_split = {**dict(test_makup), **original_val_split}  # merge two dicts

    # Split the remaining to have test, train: 15%, 70%
    train_split, val_split = train_test_split(train_val,
                                              train_size=train_size,
                                              random_state=SEED,
                                              shuffle=True)

    return dict(train_split), dict(val_split), test_split


def buil_vocab(captions: List[chain]) -> Vocabulary:
    all_words = list(chain.from_iterable(captions))
    return Vocabulary(all_words)


def create_input_arrays(
        dataset: Tuple[str, Captions],
        vocab: Vocabulary) -> Tuple[NDArray, List[List[int]], List[int]]:
    """load images and encode captions text"""

    image = load_images(dataset[0], 256, 256)
    captions_encoded, lengthes = encode_captions(dataset[1]["captions"], vocab)

    return image, captions_encoded, lengthes


def run_create_arrays(
    dataset: ImagesAndCaptions,
    vocab: Vocabulary,
    split: str,
    num_proc: int = 4
) -> Tuple[NDArray, List[List[List[int]]], List[List[int]]]:

    # Prepare arrays: images, captions encoded and captions lenghtes
    f = partial(create_input_arrays, vocab=vocab)
    num_proc = mp.cpu_count()
    with mp.Pool(processes=num_proc) as pool:
        arrays = list(
            tqdm(pool.imap(f, dataset.items()),
                 total=len(dataset),
                 desc=f"Preparing {split} Dataset",
                 unit="Image"))

    images, captions_encoded, lengthes = zip(*arrays)

    return np.stack(images), captions_encoded, lengthes
