from typing import Dict, Union

import torch
from torch import nn

from .encoder import UnetEncoder
from .heads import DescriptorHead, DetectorHead
from .keypoint_decoder import KeypointDecoder


class SuperPointNet(nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(
        self,
        encoder: Union[dict, nn.Module],
        detector_head: Union[dict, nn.Module],
        keypoint_decoder: Union[dict, None] = None,
        descriptor_head: Union[dict, nn.Module, None] = None,
        feature_channels: int = 128,
        has_dustbin: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if encoder is nn.Module:
            self.add_module("encoder", encoder)
        else:
            self.encoder = UnetEncoder(out_channels=feature_channels, **encoder)

        if detector_head is nn.Module:
            self.add_module("detector_head", detector_head)
        else:
            self.detector_head = DetectorHead(
                in_channels=feature_channels, has_dustbin=has_dustbin, **detector_head
            )

        if descriptor_head is nn.Module:
            self.add_module("descriptor_head", descriptor_head)
        elif isinstance(descriptor_head, dict):
            self.descriptor_head = DescriptorHead(
                in_channels=feature_channels, has_dustbin=has_dustbin, **descriptor_head
            )
        else:
            self.descriptor_head = None

        if keypoint_decoder is None:
            keypoint_decoder = {}
        self.keypoint_decoder = KeypointDecoder(**keypoint_decoder)

    def forward(self, x):
        feat = self.encoder(x)
        raw_detection = self.detector_head(feat)
        if self.descriptor_head:
            raw_descriptors = self.descriptor_head(feat)
        else:
            raw_descriptors = None
        return {"semi": raw_detection, "desc": raw_descriptors}

    def post_process(self, output: Dict[str, torch.Tensor]):
        # from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        semi = output["semi"]
        desc = output["desc"]
        # Flatten [batch_size, 1, H, W]
        heatmap = self.detector_head.compute_heatmap(semi)
        # nms
        heatmap_nms_batch = self.keypoint_decoder.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = self.keypoint_decoder.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs["pred"]
        # extract points
        outs = self.keypoint_decoder.batch_extract_features(
            desc, heatmap_nms_batch, residual
        )

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = None
        return output
