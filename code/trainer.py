from typing import Dict, List, Union
from statistics import mean
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import Tensor
from torch import nn

from nlg_metrics import Metrics
from utils.custom_types import ModelType, OptimType, DeviceTye, DataIterType
from utils.train_utils import seed_everything


class TrackMetrics:

    def __init__(self) -> None:

        self.reset_running()
        self.metrics = self.init_metrics()

    def create_default_dict(self):

        metrics_dict = {
            "train": defaultdict(list, {}),
            "val": defaultdict(list, {})
        }

        return metrics_dict

    def reset_running(self):
        self.running = self.create_default_dict()

    def init_metrics(self):
        return self.create_default_dict()

    def update_running(self, metrics: Dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            self.running[phase][name].append(value)

    def update(self, phase: str):
        for name, values in self.running[phase].items():
            self.metrics[phase][name].append(mean(values))
        self.reset_running()


class Trainer():

    def __init__(self,
                 optims: List[OptimType],
                 device: DeviceTye,
                 epochs: int,
                 val_interval: int,
                 stop_criteria: int,
                 pad_id: int,
                 grad_clip: float = 5.,
                 lambda_c: float = 1.,
                 resume: bool = False) -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.val_interval = val_interval
        self.stop = stop_criteria
        self.pad_id = pad_id

        # criterion and optims
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        self.img_embed_optim = optims[0]
        self.transformer_optim = optims[1]

        # metrics
        # TODO:
        # - Make as configurable parameter. Setting the Metrics class and
        # metrics tracker with it.
        # - Move tracker to Metrics class.
        # metrics functions and tracker
        self.nlgmetrics = Metrics()
        self.metrics_tracker = TrackMetrics()

        # Some coeffecient
        # cofftn of Doubly stochastic attention regularization
        self.lc = lambda_c
        self.grad_clip_c = grad_clip

    def loss_fn(self, logits: Tensor, targets: Tensor,
                attns: Tensor) -> Tensor:
        v_sz = logits.size()[-1]
        targets = targets.contiguous()
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))

        # Doubly stochastic attention regularization:
        # "Show, Attend and Tell" - arXiv:1502.03044v3 eq(14)
        # change atten size to be
        # [layer_num, head_num, batch_size, max_len, encode_size^2]
        attns = attns.permute(0, 2, 1, 3, 4)
        ln, hn = attns.size()[:2]

        # calc λ(1-∑αi)^2 for each pixel in each head in each layer
        # alphas [layer_num, head_num, batch_size*encode_size^2]
        alphas = self.lc * (1. - attns.sum(dim=3).view(ln, hn, -1))**2
        dsar = alphas.mean(-1).sum()

        return loss + dsar

    def clip_gradient(self):
        for optim in [self.img_embed_optim, self.transformer_optim]:
            for group in optim.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-self.grad_clip_c,
                                               self.grad_clip_c)

    def remove_pad(
            self, tensor: Tensor, lens: Tensor,
            mask: Tensor) -> Union[List[List[List[str]]], List[List[str]]]:
        # format 3D tensor (References) to 2D tensor
        lens = lens.view(-1)
        max_len = tensor.size(1)
        is3d = len(tensor.size()) == 3
        if is3d:
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, max_len)
            mask = mask.permute(0, 2, 1).contiguous().view(-1, max_len)

        # Remove pads: select elements that are not equal to pad (into 1d
        # tensor) then split the formed 1d tensor according to lengthes
        tensor = torch.masked_select(tensor, mask=mask)
        tensor = torch.split(tensor, split_size_or_sections=lens.tolist())
        tensor = [[str(e.item()) for e in t] for t in tensor]

        # Get back into 3d (list of list of list)
        if is3d:
            tensor = [tensor[i:i + 5] for i in range(0, len(tensor), 5)]

        return tensor

    def get_metrics(self, gtruth: Tensor, lens: Tensor, preds: Tensor):
        # gtruth [B, lm - 1, cn=5]
        # lens [B, cn=5]
        # preds [B, lm]
        mask = gtruth != self.pad_id  # mask pad tokens
        refs = self.remove_pad(gtruth, lens, mask)
        hypos = self.remove_pad(preds, lens[:, 0], mask[:, :, 0])

        scores = self.nlgmetrics.calculate(refs, hypos, self.train)

        return scores

    def set_phase(self) -> None:
        if not self.train:
            self.train = True  # toggle if val
        else:
            # validate every "val_interval" epoch
            self.train = bool(self.epoch % self.val_interval)

    def run(self, img_embeder: ModelType, transformer: ModelType,
            data_iters: DataIterType, SEED: int):
        # Sizes:
        # batch_size: B
        # image encode size^2: image seq len: is=196
        # vocab_size: vsz
        # max_len: lm=52
        # number of captions: cn=5
        # number of heads: hn=8
        # number of layers: ln

        # some preparations:
        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        img_embeder = img_embeder.to(self.device)  # move to device
        transformer = transformer.to(self.device)

        # ---------------- start epochs looping ---------------- #
        while self.epoch <= self.epochs_num:

            if self.train:
                img_embeder.train()
                transformer.train()
                data_iter = data_iters[0]
            else:
                img_embeder.eval()
                transformer.eval()
                data_iter = data_iters[1]

            # ---------------- Iterate over data ---------------- #
            # Init progress bar
            pb_trn = tqdm(data_iter, leave=False, total=len(data_iter))
            pb_trn.unit = "step"
            for step, (imgs, cptns_all, lens) in enumerate(pb_trn):
                imgs: Tensor  # images [B, 3, 256, 256]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                lens: Tensor  # lengthes of all captions [B, cn]

                # set progress bar description and metrics
                pb_trn.set_description(f"Train: Step-{step:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                cptns = cptns_all[:, :, 0].to(self.device)  # [B, lm]

                # zero the parameter gradients
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    # embed images using CNN then get logits prediction using
                    # the transformer
                    imgs = img_embeder(imgs)
                    logits, attns = transformer(imgs, cptns[:, :-1])
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, hm, B, lm, is]

                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:], attns)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                # get predections then alculate some metrics
                preds = torch.argmax(logits, dim=2).cpu()  # predections
                targets = cptns_all[:, 1:]  # remove <SOS>
                scores = self.get_metrics(targets, lens - 1, preds)
                scores["loss"] = loss.item()  # add loss to metrics scores
                self.metrics_tracker.update_running(scores, phases[self.train])

                # step ended
                # update progress bar
                pb_trn.update(1)

            # epoch ended
            self.metrics_tracker.update(phases[self.train])  # save metrics
            self.set_phase()
            self.epoch += 1 * self.train  # increase epoch count while training
