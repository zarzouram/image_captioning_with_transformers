from typing import Tuple
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class CNNFeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int,
                 dropout: float):
        super(CNNFeedForward, self).__init__()
        """
        param:
        encode_size:        encoded image size.
                            int

        embed_dim:          encoded images features dimension.
                            int

        feedforward_dim:    feedforward network model features dimension.
                            int

        dropout:            dropout value
                            float
        """
        #  Two fc layers can also be described by two cnn with kernel_size=1.
        self.conv1 = nn.Conv1d(in_channels=encode_size,
                               out_channels=feedforward_dim,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim,
                               out_channels=encode_size,
                               kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        param:
        inputs: Output from multi head attension layere.
                Tensor [batch_size, encode_size^2=196, embed_dim=512]

        output: output tensor.
                Tensor [batch_size, encode_size^2=196, embed_dim=512]
        """
        residual = inputs  # type: Tensor
        output = self.conv2(self.relu(self.conv1(inputs)))
        output = self.dropout(output)  # type: Tensor
        return self.layer_norm(output + residual)


class EncSelfAttension(nn.Module):

    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(EncSelfAttension, self).__init__()
        """
        param:
        img_embed_dim:  encoded images features dimension.
                        int

        num_heads:      number of heads in the multiheadattention model.
                        int

        dropout:        dropout value
                        float
        """
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        """
        enc_inputs:     Input images to encode
                        Tensor
                        [batch_size, encode_size * encode_size, embed_dim]

        outputs:
        enc_outputs:    Encoded image
                        Tensor
                        [batch_size, encode_size * encode_size, embed_dim]

        attn:           Attension weights
                        [batch_size, encode_size^2=196, encode_size^2=196]
        """

        residual = enc_inputs

        # enc_inputs: [image_encode_size^2=196, batch_size, img_embed_dim=512]
        enc_inputs = enc_inputs.permute(1, 0, 2)  # type: Tensor
        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs,
                                              enc_inputs)
        enc_outputs = enc_outputs.permute(1, 0, 2) + residual  # type: Tensor
        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs


class EncoderLayer(nn.Module):

    def __init__(self, img_encode_size: int, img_embed_dim: int,
                 num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        """
        param:
        img_embed_dim:  encoded images features dimension.
                        int

        num_heads:      number of heads in the multiheadattention model.
                        int

        dropout:        dropout value
                        float
        """

        self.enc_self_attn = EncSelfAttension(img_embed_dim=img_embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        self.cnn_ff = CNNFeedForward(encode_size=img_encode_size,
                                     embed_dim=img_embed_dim,
                                     feedforward_dim=512,
                                     dropout=dropout)

    def forward(self, enc_inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        enc_inputs:     Input images to encode
                        Tensor
                        [batch_size, encode_size^2=196, embed_dim=512]

        outputs:
        enc_outputs:    Encoded image
                        Tensor
                        [batch_size, encode_size^2=196, embed_dim=512]

        attn:           Attension weights
                        [batch_size, encode_size^2=196, encode_size^2=196]
        """

        enc_outputs = self.enc_self_attn(enc_inputs)
        enc_outputs = self.cnn_ff(enc_outputs)
        return enc_outputs


if __name__ == "__main__":
    import torch

    src_img = torch.rand(10, 196, 512)  # B, encode, embed
    m_test = EncoderLayer(196, 512, 8, 0.1)
    valus = m_test(src_img)
    print(valus.size())
