from typing import Dict, List, Union, Optional
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch import nn

from nlg_metrics import Metrics
from utils.custom_types import ModelType, OptimType, DeviceTye, DataIterType
from utils.custom_types import SchedulerType
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
                 schedulers: List[SchedulerType],
                 device: DeviceTye,
                 epochs: int,
                 val_interval: int,
                 early_stop: int,
                 lr_patience: int,
                 embedings_finetune: int,
                 grad_clip: float,
                 lambda_c: float,
                 checkpoints_path: str,
                 pad_id: int,
                 resume: Optional[str] = None) -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.resume = resume
        self.epochs_num = epochs
        self.epoch = 0
        self.val_interval = val_interval  # validate the model evey (n) epochs
        # stop trianing if the model doesn't improve for n-validation epochs
        self.stop = early_stop
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        self.lr_patience = lr_patience
        # start tune embeddings after n training epochs have beed passed
        self.finetune_embedding = embedings_finetune
        self.pad_id = pad_id
        self.checkpoints_path = checkpoints_path

        # criterion, optims and schedulers
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        self.img_embed_optim = optims[0]
        self.transformer_optim = optims[1]
        self.image_scheduler = schedulers[0]
        self.transformer_scheduler = schedulers[1]

        # metrics
        # TODO:
        # - Make as configurable parameter. Setting the Metrics class and
        # metrics tracker with it.
        # - Move tracker to Metrics class.
        # metrics functions and tracker
        self.nlgmetrics = Metrics()
        self.metrics_tracker = TrackMetrics()
        self.best_metric = 0

        # Some coeffecient
        # coeffecient of Doubly stochastic attention regularization
        self.lc = lambda_c
        self.grad_clip_c = grad_clip  # gradient clip coeffecient

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
        ln, hn = attns.size()[:2]  # number of layers, number of heads

        # calc λ(1-∑αi)^2 for each pixel in each head in each layer
        # alphas [layer_num, head_num, batch_size*encode_size^2]
        alphas = self.lc * (1. - attns.sum(dim=3).view(ln, hn, -1))**2
        alphas: Tensor
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
            self.train = bool(self.epoch % (self.val_interval + 1))

    def check_improvement(self, metric: float):
        is_better = metric > self.best_metric
        reduce_lr = False
        es = False
        if is_better:
            self.best_metric = metric
            self.bad_epochs_num = 0
        else:
            self.bad_epochs_num += 1

        if self.bad_epochs_num > self.lr_patience:
            reduce_lr = True
            self.num_bad_epochs = 0

        if self.bad_epochs_num > self.stop:
            es = True  # early stop

        return is_better, reduce_lr, es

    def load_checkpoint(self):
        load_path = str(Path(self.checkpoints_path) / self.resume)

        # load checkopoint
        state = torch.load(load_path, map_location=torch.device("cpu"))
        image_model_state = state["models"][0]
        transformer_state = state["models"][1]
        image_optim_state = state["optims"][0]
        transformer_optim_state = state["optims"][1]
        image_scheduler_state = state["schedulers"][0]
        transformer_scheduler_state = state["schedulers"][1]

        # load state dicts
        self.img_embed_optim.load_state_dict(image_optim_state)
        self.transformer_optim.load_state_dict(transformer_optim_state)
        self.image_scheduler.load_state_dict(image_scheduler_state)
        self.transformer_scheduler.load_state_dict(transformer_scheduler_state)

        # set some parameters
        self.train = state["phase"]
        self.epoch = state["epoch"]
        self.bad_epochs_num = state["bad_epochs_num"]
        self.best_metric = state["best_metric"]
        self.metrics_tracker.running = state["running_metrics"]
        self.metrics_tracker.metrics = state["metrics"]

        self.set_phase()  # set train or vall phase
        self.epoch += 1 * self.train

        return image_model_state, transformer_state

    def save_checkpoint(self, models: List[ModelType], save_dir: str,
                        is_best: bool):

        image_model_state = models[0].state_dict()
        Transformer_state = models[1].state_dict()
        image_optim_state = self.img_embed_optim.state_dict()
        transformer_optim_state = self.transformer_optim.state_dict()
        image_scheduler_state = self.image_scheduler.state_dict()
        transformer_scheduler_state = self.transformer_scheduler.state_dict()

        state = {
            "models": [image_model_state, Transformer_state],
            "optims": [image_optim_state, transformer_optim_state],
            "schedulers": [image_scheduler_state, transformer_scheduler_state],
            "phase": self.train,
            "epoch": self.epoch,
            "bad_epochs_num": self.bad_epochs_num,
            "best_metric": self.best_metric,
            "running_metrics": self.metrics_tracker.running,
            "metrics": self.metrics_tracker.metrics
        }

        # set save path
        file_name = "checkpoint"
        if is_best:
            file_name = f"{file_name}_best"
        save_path = Path(save_dir) / f"{file_name}.pth.tar"

        torch.save(state, save_path)

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
        phases = ["val", "train"]
        seed_everything(SEED)
        if self.resume:
            model_state_dicts = self.load_checkpoint()
            img_embeder.load_state_dict(model_state_dicts[0])
            transformer.load_state_dict(model_state_dicts[1])

        # move models to device
        img_embeder = img_embeder.to(self.device)
        transformer = transformer.to(self.device)

        # start
        main_pb = tqdm(range(self.epochs_num))
        while self.epoch <= self.epochs_num:

            main_pb.set_description(f"epoch: {self.epoch:02d}")

            is_best = False
            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:
                img_embeder.train()
                transformer.train()
                data_iter = data_iters[0]
                # fine tune the embeddings layer after some epochs and add the
                # parameters to the optimizer
                if self.epoch == self.finetune_embedding:
                    for p in transformer.decoder.cptn_emb.parameters():
                        p.requires_grad = True
                        self.transformer_optim.add_param_group({"params": p})
            else:
                img_embeder.eval()
                transformer.eval()
                data_iter = data_iters[1]

            # Iterate over data
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"
            for step, (imgs, cptns_all, lens) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 256, 256]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                lens: Tensor  # lengthes of all captions [B, cn]

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step+1:<4d}")

                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                idx = np.random.randint(0, cptns_all.size(-1))
                cptns = cptns_all[:, :, idx].to(self.device)

                # zero the parameter gradients
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    # embed images then get captions prediction
                    imgs = img_embeder(imgs)
                    logits, attns = transformer(imgs, cptns[:, :-1])
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, hm, B, lm, is]

                    loss = self.loss_fn(logits, cptns[:, 1:], attns)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                # Calculate some metrics
                preds = torch.argmax(logits, dim=2).cpu()  # predections
                targets = cptns_all[:, 1:]  # remove <SOS>
                scores = self.get_metrics(targets, lens - 1, preds)
                scores["loss"] = loss.item()  # add loss to metrics scores
                self.metrics_tracker.update_running(scores, phases[self.train])

                # step ended
                # update progress bar
                pb.update(1)

                if step == 3:
                    break

            self.metrics_tracker.update(phases[self.train])  # save metrics
            if not self.train:
                checked_metric = self.metrics_tracker.metrics["val"]["bleu4"]
                is_best, lr_r, es = self.check_improvement(checked_metric[-1])

                if lr_r:  # reduce lr
                    self.image_scheduler.step()
                    self.transformer_scheduler.step()

            # save checkpoint
            if self.train or is_best:
                self.save_checkpoint(models=[img_embeder, transformer],
                                     save_dir=self.checkpoints_path,
                                     is_best=is_best)

            # epoch ended
            self.set_phase()  # set train or vall phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            main_pb.update(1)
            if es:  # early stopping
                break
