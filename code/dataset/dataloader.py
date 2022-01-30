from typing import Tuple
import h5py
import json
import os

import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import ConstantPad1d
from torch.nn.utils.rnn import pad_sequence
# from torchvision.transforms import transforms
from torch.utils import data


class HDF5Dataset(data.Dataset):

    def __init__(self,
                 hdf5_path: str,
                 captions_path: str,
                 lengthes_path: str,
                 pad_id: float,
                 transform=None):
        super().__init__()

        self.pad_id = pad_id

        with h5py.File(hdf5_path) as h5_file:
            self.images_nm, = h5_file.keys()
            self.images = np.array(h5_file[self.images_nm])

        with open(captions_path, 'r') as json_file:
            self.captions = json.load(json_file)

        with open(lengthes_path, 'r') as json_file:
            self.lengthes = json.load(json_file)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        # get data
        # Images
        X = torch.as_tensor(self.images[i], dtype=torch.float) / 255.
        if self.transform:
            X = self.transform(X)

        # Captions: select random caption and rearrange to have it in idx=0
        # [seq_len_max, captns_num=5]
        y = [torch.as_tensor(c, dtype=torch.long) for c in self.captions[i]]
        y = pad_sequence(y, padding_value=self.pad_id)  # type: Tensor
        # # select random
        # idx = np.random.randint(0, y.size(-1))
        # y_selected = y[:, idx].view(-1, 1)
        # y = torch.hstack([y_selected, y[:, :idx], y[:, idx + 1:]])

        # Lengthes: select the random length and rearrange to have it in idx=0
        ls = torch.as_tensor(self.lengthes[i], dtype=torch.long)
        # ls_selected = ls[idx]
        # ls = torch.hstack([ls_selected, ls[:idx], ls[idx + 1:]])

        return X, y, ls

    def __len__(self):
        return self.images.shape[0]


class collate_padd(object):

    def __init__(self, max_len, pad_id=0):
        self.max_len = max_len
        self.pad = pad_id

    def __call__(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        X, y, ls = zip(*batch)
        X: Tuple[Tensor]
        y: Tuple[Tensor]
        ls: Tuple[Tensor]

        # pad tuple
        # [B, max_seq_len, captns_num=5]
        ls = torch.stack(ls)  # (B, num_captions)
        y = pad_sequence(y, batch_first=True, padding_value=self.pad)

        # pad to the max len
        pad_right = self.max_len - y.size(1)
        if pad_right > 0:
            # [B, captns_num, max_seq_len]
            y = y.permute(0, 2, 1)  # type: Tensor
            y = ConstantPad1d((0, pad_right), value=self.pad)(y)
            y = y.permute(0, 2, 1)  # [B, max_len, captns_num]

        X = torch.stack(X)  # (B, 3, 256, 256)

        return X, y, ls


if __name__ == "__main__":
    from utils import seed_worker
    from tqdm import tqdm
    from pathlib import Path

    SEED = 9001
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(SEED)

    apath = Path("/srv/data/guszarzmo/mlproject/data/mscoco_h5/")
    for p in ["train", "val", "test"]:
        img_p = str(apath / f"{p}_images.hdf5")
        cap_p = str(apath / f"{p}_captions.json")
        ls_p = str(apath / f"{p}_lengthes.json")
        train = HDF5Dataset(img_p, cap_p, ls_p, 0)

        loader_params = {
            "batch_size": 100,
            "shuffle": True,
            "num_workers": 4,
            "worker_init_fn": seed_worker,
            "generator": g
        }
        data_loader = data.DataLoader(train,
                                      collate_fn=collate_padd(30),
                                      **loader_params)

        for X, y, ls in tqdm(data_loader, total=len(data_loader)):
            pass

    print("done")
