from torch import nn


class MaskedConv2d(nn.Conv2d):
    """
      Masked convolution for pixelcnn model
    """

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        # check whether correct mask type is defined
        assert mask_type in {'A', 'B'}

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


def maskAConv(c_in=1, c_out=64, k_size=7, stride=1, pad=3):
    """2D Masked Convolution (type A)"""

    return nn.Sequential(
        MaskedConv2d('A', c_in, c_out, k_size, stride, pad, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(True))


class MaskBConvBlock(nn.Module):
    def __init__(self, h=64, k_size=7, stride=1, pad=3, residual_connection=False):
        """1x1 Conv + 2D Masked Convolution (type B) + 1x1 Conv"""

        super(MaskBConvBlock, self).__init__()

        self.residual_connection = residual_connection

        self.net = nn.Sequential(
            MaskedConv2d('B', h, h, k_size, stride, pad, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Try residual connection
        return self.net(x) + x if self.residual_connection else self.net(x)
