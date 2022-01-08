import h5py
import json
import os

import random

import numpy as np
import torch
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
            y = [torch.as_tensor(c, dtype=torch.long) for c in self.captions[i]]
            y = pad_sequence(y)
            ls = torch.as_tensor(self.lengthes[i], dtype=torch.long)

        return X, y, ls

    def __len__(self):
        return self.images.shape[0]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    '''
    X, y, ls = zip(*batch)
    y = pad_sequence(y, batch_first=True)

    X = torch.stack(X)
    ls = torch.stack(ls)

    return X, y, ls


if __name__ == "__main__":

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
                                  collate_fn=collate_fn_padd,
                                  **loader_params)

    for i in range(num_epochs):
        for j, (x, y, ls) in enumerate(data_loader):
            print(f"{i}-{j}", x.shape, y.shape, ls.shape, x.dtype, y.dtype,
                  ls.dtype)
            if j == 2:
                break
        print()
