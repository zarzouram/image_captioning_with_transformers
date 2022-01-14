from torch import nn, Tensor
import torchvision


class ImageEncoder(nn.Module):

    def __init__(self, encode_size=14, embed_dim=512):
        """
        param:
        encode_size:    encoded image size.
                        int

        embed_dim:      encoded images features dimension
                        int
        """
        super(ImageEncoder, self).__init__()

        self.embed_dim = embed_dim
        # pretrained ImageNet ResNet-101
        # Remove last linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.downsampling = nn.Conv2d(in_channels=2048,
                                      out_channels=embed_dim,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # Resize images, use 2D adaptive max pooling
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)

    def forward(self, images: Tensor):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, encode_size * encode_size, embed_dim]
        """
        # batch_size = B
        # image_size = [B, 3, h, w]
        B = images.size()[0]

        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        out = self.resnet(images)  # type: Tensor

        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))

        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        #   [B, embed_size=512, 8, 8] ->
        #       [B, embed_size=512, encode_size=14, encode_size=14] ->
        #           [B, 512, 196] -> [B, 196, 512]
        out = self.adaptive_resize(out)
        out = out.view(B, self.embed_dim, -1).permute(0, 2, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the tuning for blocks 2 through 4.
        """

        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
