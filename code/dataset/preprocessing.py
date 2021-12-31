from pathlib import Path

import json


def preprocessing_mscoco(base_dir: str, split: str, json_file: str,
                         images_dir: str, out_path: str):
    """Create h5 dataset that have two groups "image, captions" .
    The function expects to get the base directory that have mscoco jsons files
    and images folder. Also. it expect to get the split name.

    base_dir:   str
                The directory that have mscoco datset

    split:      str
                split name. train, val, test

    json_file:  str
                Path for the json annotation file, relative to base_dir

    images_dir: str
                Folder path the hold the images, relative to base dir

    out_dir:    str
                Path to wtite the h5 file
    """

    base_dir = Path(base_dir).resolve()
    json_path = base_dir / json_file
    image_dir = base_dir / images_dir
    out_path = Path(out_path).resolve()

    with open(json_path) as json_file:
        data = json.load(json_file)
