from typing import Tuple
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int

        num_heads:  number of heads in the multiheadattention model.
                    int

        dropout:    dropout value
                    float
        """

        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        param:
        dec_inputs:     Captions to decode
                        Tensor
                        [max_len, batch_size, embed_dim]

        enc_outputs:    Encoded image to decode
                        Tensor
                        [encode_size^2=196, batch_size, embed_dim]

        tgt_mask:       Mask to ensure that decoder doesn't look at future
                        tokens from a given subsequence
                        [max_len , max_len]

        tgt_pad_mask:   Mask to ensure that decoder doesn't attend pad tokens
                        [batch_size , max_len]

        outputs:
        output:         Decoder output
                        Tensor
                        [max_len, batch_size, embed_dim]

        attn:           Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        To be able to do so, I have changed the code at
                        /.virtualenvs/<env_name>/lib/python3.8/site-packages/torch/nn/functional.py
                        line 4818 and changed
                        `return attn_output, attn_output_weights.sum(dim=1) /
                        num_heads` to be
                        `return attn_output, attn_output_weights`

        """
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns


if __name__ == "__main__":
    import torch

    src_img = torch.rand(196, 10, 512)  # B, encode, embed
    captn = torch.rand(52, 10, 512)
    m_test = DecoderLayer(512, 8, 2048, 0.1)
    valus, attns = m_test(captn, src_img, None, None)
    print(valus.size())
