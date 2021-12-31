from typing import Tuple, List
from dataset.utils import get_captions, combine_image_captions, Vocabulary
import json


def parse():
    pass


def load_annotations(json_path: str) -> Tuple[list, List[str]]:
    with open(json_path) as json_file:
        data = json.load(json_file)

    annotations = data["annotations"]
    images = data["images"]

    return annotations, images


if __name__ == "__main__":

    file_path = "/srv/data/guszarzmo/mlproject/data/mscoco_original/caption_annotations/captions_train2017.json"  # noqa: E501

    annotations, images_id = load_annotations(file_path)
    captions, words = get_captions(annotations)
    images_w_captions = combine_image_captions(images_id, captions)
    vocab = Vocabulary(words)
