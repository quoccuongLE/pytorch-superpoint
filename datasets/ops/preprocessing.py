from typing import Any, Dict

import torch

from models.ops.tensor_transforms import (reshape_pixels2superpixels,
                                          reshape_superpixels2pixels)


class FromPixel2Superpixel:
    def __init__(self, superpixel_size: int, gaussian: bool = False, has_dustbin: bool = True):
        self.superpixel_size = superpixel_size
        self.gaussian = gaussian
        self.has_dustbin = has_dustbin

    def space2depth(self, x: torch.Tensor) -> torch.Tensor:
        superpixel_tensor = reshape_pixels2superpixels(
            x, superpixel_size=self.superpixel_size
        )
        if self.has_dustbin:
            batch_size, _, Hc, Wc = superpixel_tensor.shape
            dustbin = superpixel_tensor.sum(dim=1)
            dustbin = 1 - dustbin
            dustbin[dustbin < 1.0] = 0
            superpixel_tensor = torch.cat(
                (superpixel_tensor, dustbin.view(batch_size, 1, Hc, Wc)), dim=1
            )
            ## norm
            dn = superpixel_tensor.sum(dim=1)
            superpixel_tensor = superpixel_tensor.div(torch.unsqueeze(dn, 1))
        return superpixel_tensor

    def get_masks(self, mask_2D: torch.Tensor) -> torch.Tensor:
        """2D mask is constructed into 3D (Hc, Wc) space for training

        Args:
            mask_2D (torch.Tensor): tensor [batch, 1, H, W]

        Returns:
            torch.Tensor: flattened 3D mask for training
        """
        mask_3D = reshape_pixels2superpixels(
            mask_2D, superpixel_size=self.superpixel_size
        )
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    # def __call__(self, sample: Dict[str, Any]):
    #     if self.gaussian:
    #         labels_2D = sample["labels_2D_gaussian"]
    #     else:
    #         labels_2D = sample["labels_2D"]

    #     mask_2D = sample["valid_mask"]

    #     labels_3D = self.space2depth(labels_2D).float()
    #     mask_3D_flattened = self.get_masks(mask_2D)
    #     sample.update({"labels_3D": labels_3D, "mask_3D_flattened": mask_3D_flattened})
    #     return sample
