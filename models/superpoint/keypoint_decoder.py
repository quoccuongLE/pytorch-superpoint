from typing import Dict

import numpy as np
import torch

from utils.losses import extract_patches, norm_patches, soft_argmax_2d
from utils.utils import crop_or_pad_choice


class KeypointDecoder:

    def __init__(
        self,
        out_num_points: int = 500,
        patch_size: int = 5,
        nms_dist: int = 4,
        conf_thresh: float = 0.015,
        device: str = "cuda:0",
    ):
        self.out_num_points = out_num_points
        self.patch_size = patch_size
        self.device = device
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.heatmap = None
        self.heatmap_nms_batch = None

    # @staticmethod
    def pred_soft_argmax(
        self, labels_2D: torch.Tensor, heatmap: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        outs = {}
        # extract patchess
        label_idx = labels_2D[...].nonzero()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(self.device),
            heatmap.to(self.device),
            patch_size=self.patch_size,
        )
        # norm patches
        # patches = norm_patches(patches)

        # predict offsets
        patches[patches < 0] = 1e-6
        log_patches = patches.log()
        # soft_argmax
        dxdy = soft_argmax_2d(
            log_patches, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - self.patch_size // 2

        # loss
        # outs["pred"] = dxdy
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        # outs["patches"] = patches
        return dxdy, patches

    # torch
    @staticmethod
    def sample_desc_from_points(
        coarse_desc: torch.Tensor, pts: torch.Tensor, cell_size: int = 8
    ) -> torch.Tensor:
        """
        inputs:
            coarse_desc: tensor [1, 256, Hc, Wc]
            pts: tensor [N, 2] (should be the same device as desc)
        return:
            desc: tensor [1, N, D]
        """
        # --- Process descriptor.
        samp_pts = pts.transpose(0, 1)
        H, W = coarse_desc.shape[2] * cell_size, coarse_desc.shape[3] * cell_size
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            # desc = torch.zeros((D, 0))
            desc = torch.ones((1, 1, D))
        else:
            # Interpolate into descriptor map using 2D point locations.
            # samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.0)) - 1.0
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.0)) - 1.0
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            # samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(
                coarse_desc, samp_pts, align_corners=True
            )  # tensor [batch_size(1), D, 1, N]
            # desc = desc.data.cpu().numpy().reshape(D, -1)
            # desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            desc = desc.squeeze().transpose(0, 1).unsqueeze(0)
        return desc

    # extract residual
    @staticmethod
    def ext_from_points(labels_res: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        input:
            labels_res: tensor [batch, channel, H, W]
            points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
        return:
            tensor [N, channel]
        """
        labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
        points_res = labels_res[
            points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
        ]  # tensor [N, 2]
        return points_res

    def heatmap_to_nms(
        self, heatmap: torch.Tensor, tensor: bool = False, boxnms: bool = False
    ) -> np.ndarray:
        """
        return:
          heatmap_nms_batch: np [batch, 1, H, W]
        """
        to_floatTensor = lambda x: torch.from_numpy(x).type(torch.FloatTensor)
        from utils.var_dim import toNumpy

        heatmap_np = toNumpy(heatmap)
        ## heatmap_nms
        if boxnms:
            from utils.utils import box_nms

            heatmap_nms_batch = [
                box_nms(h.detach().squeeze(), self.nms_dist, min_prob=self.conf_thresh)
                for h in heatmap
            ]  # [batch, H, W]
            heatmap_nms_batch = torch.stack(heatmap_nms_batch, dim=0).unsqueeze(1)
            # print('heatmap_nms_batch: ', heatmap_nms_batch.shape)
        else:
            heatmap_nms_batch = [
                self.heatmap_nms(h, self.nms_dist, self.conf_thresh) for h in heatmap_np
            ]  # [batch, H, W]
            heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
            heatmap_nms_batch = heatmap_nms_batch[:, np.newaxis, ...]
            if tensor:
                heatmap_nms_batch = to_floatTensor(heatmap_nms_batch)
                heatmap_nms_batch = heatmap_nms_batch.to(self.device)
        self.heatmap = heatmap
        self.heatmap_nms_batch = heatmap_nms_batch
        return heatmap_nms_batch

    @staticmethod
    def heatmap_nms(
        heatmap: np.ndarray, nms_dist: int = 4, conf_thresh: float = 0.015
    ) -> np.ndarray:
        """
        input:
            heatmap: np [(1), H, W]
        """
        # nms_dist = self.config['model']['nms']
        # conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()
        boxnms = False
        # print("heatmap: ", heatmap.shape)
        from utils.utils import getPtsFromHeatmap

        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)

        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1

        return semi_thd_nms_sample

    def batch_extract_features(
        self,
        desc: torch.Tensor,
        heatmap_nms_batch: torch.Tensor,
        residual: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # extract pts, residuals for pts, descriptors
        """
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        batch_size = heatmap_nms_batch.shape[0]

        pts_int, pts_offset, pts_desc = [], [], []
        pts_idx = heatmap_nms_batch[...].nonzero()  # [N, 4(batch, 0, y, x)]
        for i in range(batch_size):
            mask_b = pts_idx[:, 0] == i  # first column == batch
            pts_int_b = pts_idx[mask_b][:, 2:].float()  # default floatTensor
            pts_int_b = pts_int_b[:, [1, 0]]  # tensor [N, 2(x,y)]
            res_b = residual[mask_b]
            pts_b = pts_int_b + res_b  # .no_grad()
            # extract desc
            pts_desc_b = self.sample_desc_from_points(
                desc[i].unsqueeze(0), pts_b
            ).squeeze(0)

            # Get random shuffle
            choice = crop_or_pad_choice(
                pts_int_b.shape[0], out_num_points=self.out_num_points, shuffle=True
            )
            choice = torch.tensor(choice)
            pts_int.append(pts_int_b[choice])
            pts_offset.append(res_b[choice])
            pts_desc.append(pts_desc_b[choice])

        pts_int = torch.stack((pts_int), dim=0)
        pts_offset = torch.stack((pts_offset), dim=0)
        pts_desc = torch.stack((pts_desc), dim=0)
        return pts_int, pts_offset, pts_desc
