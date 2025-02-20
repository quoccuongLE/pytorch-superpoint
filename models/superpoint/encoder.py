import torch
from torch import nn


class UnetEncoder(nn.Module):

    _enc_channels = [64, 64, 128]
    _mp_layers = [True, True, True]

    def __init__(self, in_channels: int = 1, out_channels: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _blocks = []
        current_channels = in_channels
        for mp_bool, channel in zip(self._mp_layers, self._enc_channels):
            conv_block = self._block(in_channels=current_channels, features=channel)
            _blocks.extend(conv_block)
            if mp_bool:
                mp_block = nn.MaxPool2d(kernel_size=2)  # Modified from Unet (stride=2)
                _blocks.append(mp_block)
            current_channels = channel

        # Final block
        out_block = self._block(in_channels=current_channels, features=out_channels)
        _blocks.extend(out_block)

        self.layers = nn.Sequential(*_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @staticmethod
    def _block(in_channels: int, features: int) -> nn.Module:
        # ReLU layer added every Conv layer
        return [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        ]
