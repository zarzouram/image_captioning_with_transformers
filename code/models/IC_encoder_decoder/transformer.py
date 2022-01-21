from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, Tensor

from encoder_layers import EncoderLayer
from decoder_layers import DecoderLayer
from pe import PositionalEncoding


class Encoder(nn.Module):
    """
    param:

    layer:      an instance of the EecoderLayer() class

    num_layers: the number of decoder-layers
                int
    """

    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        # Make copies of the encoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        """
        param:
        x:  encoder input
            Tensor
            [encode_size^2=196, batch_size, embed_dim=512]

        outputs:
        x:  encoder output
            Tensor
            [encode_size^2=196, batch_size, embed_dim=512]
        """

        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    """
    param:
    layer:          an instance of the EecoderLayer() class

    vocab_size:     the number of vocabulary
                    int

    d_model:        size of features in the transformer inputs
                    int

    num_layers:     the number of decoder-layers
                    int

    max_len:        maximum len pf target captions
                    int

    dropout:        dropout value
                    float

    pad_id:         padding token id
                    float
    """

    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        param:
        tgt_cptn:   Captions (Transformer target sequence)
                    Tensor
                    [batch_size, max_len]

        src_img:    Encoded images (Transformer source sequence)
                    Tensor
                    [encode_size^2=196, batch_size, embed_dim=512]

        outputs:
        output:     Decoder output
                    Tensor
                    [max_len, batch_size, embed_dim=512]

        attn_all:   Attension weights
                    Tensor
                    [batch_size, head_num, max_len, encode_size^2=196]
        """

        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn != self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [head_num, B, max_len, encode_size] ->
        # [B, head_num, max_len, encode_size]
        attns_all = torch.stack(attns_all).permute(1, 0, 2, 3)

        return tgt_cptn, attns_all


class Transformer(nn.Module):
    """
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 img_encode_size: int,
                 enc_ff_dim: int,
                 dec_ff_dim: int,
                 enc_n_layers: int,
                 dec_n_layers: int,
                 enc_n_heads: int,
                 dec_n_heads: int,
                 max_len: int,
                 dropout: float = 0.1,
                 pad_id: int = 0):
        super(Transformer, self).__init__()
        encoder_layer = EncoderLayer(img_encode_size=img_encode_size,
                                     img_embed_dim=d_model,
                                     feedforward_dim=enc_ff_dim,
                                     num_heads=enc_n_heads,
                                     dropout=dropout)
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=dec_n_heads,
                                     feedforward_dim=dec_ff_dim,
                                     dropout=dropout)
        self.encoder = Encoder(layer=encoder_layer, num_layers=enc_n_layers)
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=vocab_size,
                               d_model=d_model,
                               num_layers=dec_n_layers,
                               max_len=max_len,
                               dropout=dropout,
                               pad_id=pad_id)

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def load_pretrained_embeddings(self, embeddings):
        self.decoder.cptn_emb.weight = nn.Parameter(embeddings)

    def forward(self, images: Tensor,
                captions: Tensor) -> Tuple[Tensor, Tensor]:
        # encode, decode, predict
        images_encoded = self.encoder(images.permute(1, 0, 2))
        tgt_cptn, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(tgt_cptn)

        return predictions, attns


if __name__ == "__main__":

    src_img = torch.rand(10, 196, 512)  # B, encode, embed
    captions = torch.randint(0, 52, (10, 30), dtype=torch.long)
    m_test = Transformer(52, 512, 196, 512, 2048, 2, 8, 8, 8, 30, 0.1, 0)
    valus, attns = m_test(src_img, captions)
    print(valus.size())
    print(attns.size())
