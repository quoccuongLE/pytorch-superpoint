import torch
from torch import nn

from models.ops.tensor_transforms import (
    reshape_pixels2superpixels,
    reshape_superpixels2pixels,
)


class DetectorHead(nn.Module):

    _superpixel_size = 8
    _conv_configs = [
        dict(out_channels=256, kernel_size=3, stride=1, padding=1),
    ]

    def __init__(self, in_channels: int, has_dustbin: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_dustbin = has_dustbin
        _blocks = []
        feat_channels = in_channels
        for conv_config in self._conv_configs:
            conv_config.update(dict(in_channels=feat_channels))
            _block = nn.Conv2d(**conv_config)
            feat_channels = conv_config["out_channels"]
            _bn = nn.BatchNorm2d(num_features=feat_channels)
            _relu = nn.ReLU(inplace=True)
            _blocks.extend([_block, _bn, _relu])

        # Output head
        _blocks.extend(
            [
                nn.Conv2d(
                    in_channels=feat_channels,
                    out_channels=self._superpixel_size**2 + 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(num_features=self._superpixel_size**2 + 1),
            ]
        )
        self.layers = nn.Sequential(*_blocks)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.layers(feat)

    def depth2space(self, x: torch.Tensor) -> torch.Tensor:
        return reshape_superpixels2pixels(x, superpixel_size=self._superpixel_size)

    def space2depth(self, x: torch.Tensor) -> torch.Tensor:
        superpixel_tensor =  reshape_pixels2superpixels(x, superpixel_size=self._superpixel_size)
        if self.has_dustbin:
            batch_size, _, Hc, Wc = superpixel_tensor.shape
            dustbin = superpixel_tensor.sum(dim=1)
            dustbin = 1 - dustbin
            dustbin[dustbin < 1.0] = 0
            # print('dust: ', dustbin.shape)
            # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
            superpixel_tensor = torch.cat(
                (superpixel_tensor, dustbin.view(batch_size, 1, Hc, Wc)), dim=1
            )
            ## norm
            dn = superpixel_tensor.sum(dim=1)
            superpixel_tensor = superpixel_tensor.div(torch.unsqueeze(dn, 1))
        return superpixel_tensor

    def compute_heatmap(self, feat: torch.Tensor):
        batch_size = feat.shape[0] if feat.dim() == 4 else -1

        if batch_size > 0:
            # [batch, 65, Hc, Wc]
            dense = nn.functional.softmax(feat, dim=1)
            # Remove dustbin.
            nodust = dense[:, :-1, :, :]
        else:
            dense = nn.functional.softmax(feat, dim=0)  # [65, Hc, Wc]
            nodust = dense[:-1, :, :].unsqueeze(0)
        # Reshape to get full resolution heatmap [1, H, W]
        heatmap = self.depth2space(nodust)
        heatmap = heatmap.squeeze(0) if batch_size == -1 else heatmap
        return heatmap


class DescriptorHead(nn.Module):

    _conv_configs = [
        dict(out_channels=256, kernel_size=3, stride=1, padding=1),
    ]

    def __init__(self, in_channels: int, out_channels: int = 256, has_dustbin: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_dustbin = has_dustbin
        _blocks = []
        feat_channels = in_channels
        for conv_config in self._conv_configs:
            conv_config.update(dict(in_channels=feat_channels))
            _block = nn.Conv2d(**conv_config)
            feat_channels = conv_config["out_channels"]
            _bn = nn.BatchNorm2d(num_features=feat_channels)
            _relu = nn.ReLU(inplace=True)
            _blocks.extend([_block, _bn, _relu])
        # Output head
        _blocks.extend(
            [
                nn.Conv2d(
                    in_channels=feat_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            ]
        )
        self.layers = nn.Sequential(*_blocks)
        self._out_channels = out_channels

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        output = self.layers(feat)
        norm_vecs = torch.norm(output, p=2, dim=1)  # Compute the norm.
        return output.div(torch.unsqueeze(norm_vecs, 1))  # Divide by norm to normalize.
