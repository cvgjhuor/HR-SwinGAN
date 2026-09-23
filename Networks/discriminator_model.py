import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """Initialize convolutional and batch normalization layer weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# PatchGAN discriminator predicting local realism scores across image patches
class PatchGAN(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 crop_center=None, FC_bottleneck=False):
        """Construct multi-scale convolutional PatchGAN discriminator."""
        super(PatchGAN, self).__init__()
        self.crop_center = crop_center

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        # Initial convolutional layers with progressive downsampling
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # Multi-scale feature extraction stages
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Final classification head: 1-channel patch map or optional FC bottleneck
        if FC_bottleneck:
            sequence += [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(ndf * nf_mult, 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128, 1)
            ]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            ]

        self.model = nn.Sequential(*sequence).apply(weights_init)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict patch-wise realism score map."""
        # Optional center crop to focus discrimination on central region
        if self.crop_center is not None:
            _, _, h, w = input_tensor.shape
            x0 = (h - self.crop_center) // 2
            y0 = (w - self.crop_center) // 2
            input_tensor = input_tensor[:, :, x0:x0 + self.crop_center, y0:y0 + self.crop_center]

        # Output patch score map [B, 1, H', W']
        return self.model(input_tensor)
