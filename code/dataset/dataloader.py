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
                 transform=None):
        super().__init__()

        with h5py.File(hdf5_path) as h5_file:
            self.images_nm, = h5_file.keys()
            self.images = np.array(h5_file[self.images_nm])

        with open(captions_path, 'r') as json_file:
            self.captions = json.load(json_file)

        with open(lengthes_path, 'r') as json_file:
            self.lengthes = json.load(json_file)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def __getitem__(self, i):
        # get data
        # Images
        X = torch.as_tensor(self.images[i], dtype=torch.float) / 255.
        if self.transform:
            X = self.transform(X)

        if self.images_nm == "train":
            # Captions and Lengthes
            captn_id = np.random.randint(0, 5)
            y = torch.as_tensor(self.captions[i][captn_id], dtype=torch.long)
            ls = torch.as_tensor(self.lengthes[i][captn_id], dtype=torch.long)

        else:
            y = [
                torch.as_tensor(c, dtype=torch.long) for c in self.captions[i]
            ]
            y = pad_sequence(y)
            ls = torch.as_tensor(self.lengthes[i], dtype=torch.long)

        return X, y, ls

    def __len__(self):
        return self.images.shape[0]


class collate_padd(object):

    def __init__(self, max_len, device, pad_id=0.):
        self.max_len = max_len
        self.pad = pad_id
        self.device = device

    def __call__(self, batch):
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        X, y, ls = zip(*batch)

        y = pad_sequence(y, batch_first=True)  # type: Tensor
        pad_right = self.max_len - y.size()[1]
        if pad_right < 0:  # truncate if len > max_len
            y = torch.hstack((y[:, :self.max_len - 1], y[:, -1].unsqueeze(1)))
        elif pad_right > 0:  # pad to the max_len
            if len(y.size()) == 3:  # test and val split y = [B, max_seq_len, 5]
                y = y.permute(0, 2, 1)  # [B, 5, max_seq_len]
            y = ConstantPad1d((0, pad_right), value=self.pad)(y)

        y = y.to(self.device)
        X = torch.stack(X).to(self.device)
        ls = torch.stack(ls).to(self.device)
        if len(y.size()) == 3:  # change back the y size [B,  max_seq_len, 5]
            y = y.permute(0, 2, 1)

        return X, y, ls


if __name__ == "__main__":
    from utils import seed_worker

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

    img_p = "/srv/data/guszarzmo/mlproject/data/mscoco_h5/train_images.hdf5"
    cap_p = "/srv/data/guszarzmo/mlproject/data/mscoco_h5/train_captions.json"
    ls_p = "/srv/data/guszarzmo/mlproject/data/mscoco_h5/train_lengthes.json"
    train = HDF5Dataset(img_p, cap_p, ls_p)

    num_epochs = 2
    loader_params = {
        "batch_size": 3,
        "shuffle": True,
        "num_workers": 4,
        "worker_init_fn": seed_worker,
        "generator": g
    }
    data_loader = data.DataLoader(train,
                                  collate_fn=collate_padd(30),
                                  **loader_params)

    for i in range(num_epochs):
        for j, (x, y, ls) in enumerate(data_loader):
            print(f"{i}-{j}", x.shape, y.shape, ls.shape, x.dtype, y.dtype,
                  ls.dtype)
            if j == 2:
                break
        print()
