from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose

from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

from dataset.dataloader import HDF5Dataset, collate_padd
from dataset.vocab import Vocabulary
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device


def get_datasets(dataset_dir: str):
    # Setting some pathes
    dataset_dir = Path(dataset_dir)
    images_train_path = dataset_dir / "train_images.hdf5"
    images_val_path = dataset_dir / "val_images.hdf5"
    captions_train_path = dataset_dir / "train_captions.json"
    captions_val_path = dataset_dir / "val_captions.json"
    lengthes_train_path = dataset_dir / "train_lengthes.json"
    lengthes_val_path = dataset_dir / "val_lengthes.json"

    # images transfrom
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    train_dataset = HDF5Dataset(hdf5_path=images_train_path,
                                captions_path=captions_train_path,
                                lengthes_path=lengthes_train_path,
                                transform=transform)

    val_dataset = HDF5Dataset(hdf5_path=images_val_path,
                              captions_path=captions_val_path,
                              lengthes_path=lengthes_val_path,
                              transform=transform)

    return train_dataset, val_dataset


if __name__ == "__main__":

    # parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # mscoco hdf5 and json files
    resume = args.resume

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load confuguration file
    config = load_json(args.config_path)

    # load vocab
    vocab = Vocabulary()
    vocab.load_vocab(str(Path(dataset_dir) / "vocab.json"))

    # SEED
    SEED = 9001
    seed_everything(SEED)

    # --------------- dataloader --------------- #
    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]
    train_ds, val_ds = get_datasets(dataset_dir)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_padd(max_len, device),
                            pin_memory=True,
                            **loader_params)
    train_iter = DataLoader(val_ds,
                            collate_fn=collate_padd(max_len, device),
                            batch_size=1,
                            pin_memory=True,
                            num_workers=1,
                            shuffle=True)
    print("loading dataset finished.\n")

    # --------------- Construct models, optimizers --------------- #
    print("Construct models...")
    # prepare some hyperparameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab.__len__()
    transformer_hyperparms["pad_id"] = vocab.stoi["<pad>"]
    img_encode_size = int(image_enc_hyperparms["encode_size"]**2)
    transformer_hyperparms["img_encode_size"] = img_encode_size
    transformer_hyperparms["max_len"] = max_len
    # construct models
    image_enc = ImageEncoder(**image_enc_hyperparms).to(device)
    image_enc.fine_tune(True)
    transformer = Transformer(**transformer_hyperparms).to(device)
    # Optimizer
    image_enc_lr = config["train_parms"]["encoder_lr"]
    transformer_lr = config["train_parms"]["transformer_lr"]
    parms2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    image_encoder_optim = torch.optim.Adam(params=parms2update,
                                           lr=image_enc_lr)
    parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
    transformer_optim = torch.optim.Adam(params=parms2update,
                                         lr=transformer_lr)

    print("done")
