import os
from dataset.custom_types import ImagesAndCaptions
from argparse import Namespace

from pathlib import Path
from itertools import chain

import torch

from utils.train_utils import seed_everything
from dataset.utils import parse_arguments, load_json, write_h5_dataset
from dataset.utils import write_json
from dataset.dataset_helper import get_captions, combine_image_captions
from dataset.dataset_helper import run_create_arrays
from dataset.dataset_helper import split_dataset, build_vocab


def get_data(json_path: str,
             imgs_dir: str,
             max_len: int = 52) -> ImagesAndCaptions:
    """Load annations json file and return a images ids with its captions in
        the following format:
            image_name: {image_id: list of captions tokens}
    """

    annotations, images_id = load_json(json_path)
    captions = get_captions(annotations, max_len)
    images_w_captions = combine_image_captions(images_id, captions, imgs_dir)

    return images_w_captions


if __name__ == "__main__":

    SEED = 9001
    seed_everything(seed=SEED)

    # parse argument command
    args = parse_arguments()  # type: Namespace

    # process some directories
    ds_dir = Path(os.path.expanduser(args.dataset_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))
    train_ann_path = str(ds_dir / args.json_train)  # train annotations path
    val_ann_path = str(ds_dir / args.json_val)  # val annotations path
    train_imgs_dir = str(ds_dir / args.image_train)  # train images path
    val_imgs_dir = str(ds_dir / args.image_val)  # val images path
    output_dir.mkdir(parents=True, exist_ok=True)
    # torchtext.vocab
    # Vector must be one of the vectors supoorted by
    # torchtext.vocab.Vectors classes
    # https://github.com/pytorch/text/blob/0169cde2f1d446ae886ef0be07e9a673585ed256/torchtext/vocab.py#L151

    vector_dir = Path(os.path.expanduser(args.vector_dir))
    vector_name = list(vector_dir.glob("*.zip"))  # dir must have one zip file
    vector_name = f"{vector_name[0].name.strip('.zip')}.{args.vector_dim}d"

    # process annotation files
    print("Process annotation files...")
    images_captions = get_data(train_ann_path, train_imgs_dir, args.max_len)
    images_captions_test = get_data(val_ann_path, val_imgs_dir)

    # split data
    train_ds, val_ds, test_ds = split_dataset(images_captions,
                                              images_captions_test, SEED)

    # Create vocab from train dataset set OOV to <UNK>, then encode captions
    captions = [chain.from_iterable(d["captions"]) for d in train_ds.values()]
    vocab = build_vocab(captions, str(vector_dir), vector_name, args.min_freq)
    print("Processing finished.\n")

    # Create numpy arrays for images, list of list of list of str for captions
    # after encoding them and list of list for captions lengthes then save them
    for ds, split in zip([train_ds, val_ds, test_ds],
                         ["train", "val", "test"]):
        # create arrays
        arrs = run_create_arrays(dataset=ds, vocab=vocab, split=split)
        images, captions_encoded, lengthes = arrs
        print(f"Number of samples in the {split} split:   {images.shape[0]}")

        # saving dataset
        print(f"Saving {split} dataset ...")
        write_h5_dataset(write_path=str(output_dir / f"{split}_images.hdf5"),
                         name=split,
                         data=images,
                         type="uint8")

        write_json(str(output_dir / f"{split}_captions.json"),
                   captions_encoded)
        write_json(str(output_dir / f"{split}_lengthes.json"), lengthes)
        print(f"Saving {split} dataset finished.\n")

    torch.save(vocab, str(output_dir / "vocab.pth"))

print("\nCreating dataset files finished.\n")
