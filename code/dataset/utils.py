from typing import DefaultDict, List, Dict, Tuple
from argparse import Namespace

import argparse
from collections import defaultdict
import json
import re

captions = DefaultDict[str, List[List[str]]]
images_w_captions = Dict[str, captions]


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

    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU device to be used")

    args = parser.parse_args()

    return args


def load_json(json_path: str) -> Tuple[list, List[str]]:
    with open(json_path) as json_file:
        data = json.load(json_file)

    annotations = data["annotations"]
    images = data["images"]

    return annotations, images


def get_captions(annotations: list) -> captions:
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
