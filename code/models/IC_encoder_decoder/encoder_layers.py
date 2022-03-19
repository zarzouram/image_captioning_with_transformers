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
        # TODO:
        # Need to be revisited. Not correct!
        # Based on:
        # https://github.com/RoyalSkye/Image-Caption/blob/e528b36b32fdc8175921ce60bb9a2c6cecafebb8/transformer.py#L73-L93
        # Two fc layers can also be described by two cnn with kernel_size=1.
        # https://sebastianraschka.com/faq/docs/fc-to-conv.html#methods-2-convolution-with-1x1-kernels
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
                Tensor [encode_size^2, batch_size, embed_dim]

        output: output tensor.
                Tensor [encode_size^2, batch_size, embed_dim]
        """
        output = self.conv2(self.relu(self.conv1(inputs.permute(1, 0, 2))))
        output = self.dropout(output)  # type: Tensor
        return self.layer_norm(output.permute(1, 0, 2) + inputs)


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
        param:
        enc_inputs:     Input images to encode
                        Tensor
                        [encode_size^2, batch_size, embed_dim]

        outputs:
        enc_outputs:    Encoded image
                        Tensor
                        [encode_size^2, batch_size, embed_dim]
        """

        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs,
                                              enc_inputs)
        enc_outputs = enc_outputs + enc_inputs
        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs


class EncoderLayer(nn.Module):

    def __init__(self, img_encode_size: int, img_embed_dim: int,
                 feedforward_dim: int, num_heads: int, dropout: float):
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
                                     feedforward_dim=feedforward_dim,
                                     dropout=dropout)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        """
        param:
        enc_inputs:     Input images to encode
                        Tensor
                        [encode_size^2, batch_size, embed_dim]

        outputs:
        enc_outputs:    Encoded image
                        Tensor
                        [encode_size^2, batch_size, embed_dim]
        """

        enc_outputs = self.enc_self_attn(enc_inputs)
        enc_outputs = self.cnn_ff(enc_outputs)
        return enc_outputs


if __name__ == "__main__":
    import torch

    src_img = torch.rand(196, 10, 512)  # B, encode, embed
    m_test = EncoderLayer(196, 512, 512, 8, 0.1)
    valus = m_test(src_img)
    print(valus.size())
