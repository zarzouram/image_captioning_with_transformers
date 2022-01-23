import torch
from torch import nn


class Trainer():

    def __init__(self,
                 optims,
                 epochs,
                 device,
                 pad_id,
                 resume: bool = False) -> None:

        self.train = True  # train or val
        self.epochs = epochs
        self.device = device
        self.resume = resume

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        self.img_embed_optim = optims[0]
        self.transformer_optim = optims[1]

    def loss_fn(self, logits, targets):
        v_sz = logits.size()[-1]
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))

        return loss

    def run(self, img_embeder, transformer, data_iters):

        img_embeder = img_embeder.to(self.device)
        transformer = transformer.to(self.device)
        for epoch in range(self.epochs):
            if self.train:
                img_embeder.train()
                transformer.train()
                data_iter = data_iters[0]
            else:
                img_embeder.eval()
                transformer.eval()
                data_iter = data_iters[1]

            # Iterate over data.
            for (imgs, cptns, lens) in data_iter:

                imgs = imgs.to(self.device)
                cptns = cptns.to(self.device)
                lens = lens.to(self.device)

                # zero the parameter gradients
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    imgs = img_embeder(imgs)
                    logits, attns = transformer(imgs, cptns)

                    loss = self.loss_fn(logits, cptns)
                    loss.backward()

                    if self.train:
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                    print("stop")
