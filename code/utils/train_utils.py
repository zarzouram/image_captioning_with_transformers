import os
import random

from argparse import Namespace
import argparse
import json

import numpy as np
import torch


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LT2326 H21 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/srv/data/guszarzmo/mlproject/data/mscoco_h5",
                        help="Directory contains processed MS COCO dataset.")

    parser.add_argument(
        "--config_path",
        type=str,
        default="code/config.json",  # noqa: E501
        help="Path for the configuration json file.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu
        help='Device to be used either gpu or cpu.')

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help='checkpoint filename.')

    args = parser.parse_args()

    return args


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(json_path: str) -> dict:
    with open(json_path) as json_file:
        data = json.load(json_file)

    return data
