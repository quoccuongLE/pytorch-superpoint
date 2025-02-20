import torch
from torch import nn


class DetectorHead(nn.Module):

    _superpixel_size = 8
    _conv_configs = [
        dict(out_channels=256, kernel_size=3, stride=1, padding=1),
    ]

    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        output = x.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self._superpixel_size**2)
        s_width = int(d_width * self._superpixel_size)
        s_height = int(d_height * self._superpixel_size)
        t_1 = output.reshape(
            batch_size, d_height, d_width, self._superpixel_size**2, s_depth
        )
        spl = t_1.split(self._superpixel_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = (
            torch.stack(stack, 0)
            .transpose(0, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, s_height, s_width, s_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output

    def space2depth(self, x: torch.Tensor) -> torch.Tensor:
        output = x.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self._superpixel_size**2
        d_width = int(s_width / self._superpixel_size)
        d_height = int(s_height / self._superpixel_size)
        t_1 = output.split(self._superpixel_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

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

    def __init__(self, in_channels: int, out_channels: int = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
