from torch import nn, Tensor
import torchvision


class ImageEncoder(nn.Module):

    def __init__(self, image_size_encoded=14):
        super(ImageEncoder, self).__init__()

        # pretrained ImageNet ResNet-101
        # Remove last linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.downsampling = nn.Conv2d(in_channels=2048,
                                      out_channels=512,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # Resize images, use 2D adaptive max pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(image_size_encoded)

    def forward(self, images):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images
                Tensor [batch_size, h_encoded=14, w_encoded=14, 2048]
        """
        # batch_size B
        # image_size = [B, 3, h, w]

        # [B, 3, h, w] -> [B, 2048, h/32, w/32]
        out = self.resnet(images)  # type: Tensor

        # [B, 2048, 8, 8] -> [B, 512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))

        # [B, 512, 8, 8] -> [B, 512, 14, 14]
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
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
