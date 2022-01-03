from typing import List, Tuple
from argparse import Namespace
from numpy.typing import NDArray

import argparse
import json
import h5py

import numpy as np


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LT2326 H21 Mohamed's Project")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/srv/data/guszarzmo/mlproject/data/mscoco_original/",
        help="Directory contains  MS COCO dataset files.")

    parser.add_argument(
        "--json_train",
        type=str,
        default="caption_annotations/captions_train2017.json",
        help="Directory have MS COCO annotations file for the train split.")

    parser.add_argument(
        "--json_val",
        type=str,
        default="caption_annotations/captions_val2017.json",
        help="Directory have MS COCO annotations file for the val split.")

    parser.add_argument(
        "--image_train",
        type=str,
        default="images/train2017",
        help="Directory have MS COCO images files for the train split.")

    parser.add_argument(
        "--image_val",
        type=str,
        default="images/val2017",
        help="Directory have MS COCO image files for the val split.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/srv/data/guszarzmo/mlproject/data/mscoco_h5/",
        help="Directory have MS COCO image files for the val split.")

    args = parser.parse_args()

    return args


def load_json(json_path: str) -> Tuple[list, List[str]]:
    with open(json_path) as json_file:
        data = json.load(json_file)

    annotations = data["annotations"]
    images = data["images"]

    return annotations, images


def write_json(json_path: str, data) -> None:
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def write_h5_dataset(write_path: str, data: NDArray, name: str,
                     type: str) -> None:

    with h5py.File(write_path, "w") as h5f:
        h5f.create_dataset(name=name,
                           data=data,
                           shape=np.shape(data),
                           dtype=type)
