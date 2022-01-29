from pathlib import Path
from math import sqrt

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.optim import Adam

from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

from trainer import Trainer
from dataset.dataloader import HDF5Dataset, collate_padd
from dataset.vocab import Vocabulary, DictWithDefault
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device


def get_datasets(dataset_dir: str, pid_pad: float):
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
                                pad_id=pid_pad,
                                transform=transform)

    val_dataset = HDF5Dataset(hdf5_path=images_val_path,
                              captions_path=captions_val_path,
                              lengthes_path=lengthes_val_path,
                              pad_id=pid_pad,
                              transform=transform)

    return train_dataset, val_dataset


def get_glove_embedding(path_to_glove: str):
    glove_embeds = DictWithDefault(default=0)
    with open(path_to_glove) as f:
        for line in f:
            s = line.strip().split()
            glove_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    return glove_embeds


def get_glove_weights(glove_embeds: DictWithDefault, stoi: DictWithDefault,
                      vocab_size: int):

    embed_dim = list(glove_embeds.values())[-1].shape[0]
    weights = np.zeros((vocab_size, embed_dim))  # init embeddings weights
    for w_str, w_id in stoi.items():
        weight = glove_embeds[w_str]  # if OOV/UNK -> weight(dict default)=int
        if isinstance(weight, int):
            weight = glove_embeds[w_str.lower()]
            if isinstance(weight, int):
                continue

        weights[w_id] = weight

    # Initailize <UNK> token
    weight_unk = torch.ones(1, embed_dim)
    torch.nn.init.xavier_uniform_(weight_unk)
    weights[stoi.default] = weight_unk
    # initialize start and end token
    w = np.random.uniform(-sqrt(0.06), sqrt(0.06), (1, embed_dim))
    weights[stoi["<SOS>"]] = w
    weights[stoi["<EOS>"]] = w
    return torch.FloatTensor(weights)


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
    min_freq = config["min_freq"]
    vocab = Vocabulary()
    vocab.load_vocab(str(Path(dataset_dir) / "vocab.json"), min_freq)
    pad_id = vocab.stoi["<PAD>"]
    vocab_size = vocab.__len__()

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # --------------- dataloader --------------- #
    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]
    train_ds, val_ds = get_datasets(dataset_dir, pad_id)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_padd(max_len, pad_id),
                            pin_memory=True,
                            **loader_params)
    val_iter = DataLoader(val_ds,
                          collate_fn=collate_padd(max_len, pad_id),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=1,
                          shuffle=True)
    print("loading dataset finished.")
    print(f"number of vocabualry is {vocab_size}\n")

    # --------------- Construct models, optimizers --------------- #
    print("constructing models")
    # prepare some hyperparameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_hyperparms["encode_size"]**2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1

    # construct models
    image_enc = ImageEncoder(**image_enc_hyperparms)
    image_enc.fine_tune(True)
    transformer = Transformer(**transformer_hyperparms)

    # load pretrained embeddings
    print("loading pretrained glove embeddings...")
    embeds_vectors = get_glove_embedding(config["pathes"]["embedding_path"])
    weights = get_glove_weights(embeds_vectors, vocab.stoi, vocab_size)
    transformer.decoder.cptn_emb.from_pretrained(weights,
                                                 freeze=True,
                                                 padding_idx=pad_id)

    # Optimizer
    image_enc_lr = config["optim_params"]["encoder_lr"]
    parms2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    image_encoder_optim = Adam(params=parms2update, lr=image_enc_lr)

    transformer_lr = config["optim_params"]["transformer_lr"]
    parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
    transformer_optim = Adam(params=parms2update, lr=transformer_lr)

    # --------------- Training --------------- #
    print("start training...\n")
    train = Trainer(optims=[image_encoder_optim, transformer_optim],
                    device=device,
                    pad_id=pad_id,
                    **config["train_parms"])
    train.run(image_enc, transformer, [train_iter, val_iter], SEED)

    print("done")
