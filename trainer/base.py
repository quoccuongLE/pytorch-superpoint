import logging
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from models.ops.tensor_transforms import reshape_pixels2superpixels
from models.ops.utils import get_point_from_heatmap
from models.superpoint.superpoint_net import SuperPointNet
from utils.experimental.tools import dict_update
from utils.experimental.loader import load_checkpoint
from utils.utils import descriptor_loss
from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse
from utils.loader import pretrainedLoader as pretrained_loader

from .utils import get_ndarray, to_floatTensor, update_overlap

TrainerMode = Literal["train", "val", "test"]


def precision_recall_metrics(pred, labels):
    offset = 10**-6
    assert (
        pred.size() == labels.size()
    ), "Sizes of pred, labels should match when you get the precision/recall!"
    precision = torch.sum(pred * labels) / (torch.sum(pred) + offset)
    recall = torch.sum(pred * labels) / (torch.sum(labels) + offset)
    assert precision.item() <= 1.0 and precision.item() >= 0.0
    return {"precision": precision, "recall": recall}


class BaseTrainer:
    """
    # This is the base class for training classes. Wrap pytorch net to help training process.

    """

    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(
        self,
        config: dict,
        save_path: Path = Path("."),
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        ## default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]

        :param config:
            dense_loss, sparse_loss (default)

        :param save_path:
        :param device:
        :param verbose:
        """
        print("Load Train_model_heatmap!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False

        self.batch_size = self.config["model"]["batch_size"]
        self.max_iter = config["train_iter"]
        self.max_epoch = int(np.ceil(self.max_iter / self.batch_size))

        self.model_name = self.config["model"]["name"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        self.print_important_config()
        # add_dustbin = False
        # if det_loss_type == "l2":
        #     add_dustbin = False
        # elif det_loss_type == "softmax":
        #     add_dustbin = True

    def print_important_config(self):
        """
        # print important configs
        :return:
        """
        print("=" * 10, " check!!! ", "=" * 10)

        print("learning_rate: ", self.config["model"]["learning_rate"])
        print("lambda_loss: ", self.config["model"]["lambda_loss"])
        print("detection_threshold: ", self.config["model"]["detection_threshold"])
        print("batch_size: ", self.config["model"]["batch_size"])

        print("=" * 10, " descriptor: ", self.desc_loss_type, "=" * 10)
        for item in list(self.desc_params):
            print(item, ": ", self.desc_params[item])

        print("=" * 32)

    def data_parallel(self):
        """
        put network and optimizer to multiple gpus
        :return:
        """
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        # self.model = nn.DataParallel(self.model)
        self.optimizer = self.get_optimizer(
            self.model, lr=self.config["model"]["learning_rate"]
        )

    def get_optimizer(self, model: nn.Module, lr: float):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("ADAM optimizer")
        return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    def load_model(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        # model_name = self.config["model"]["name"]
        params = self.config["model"]["params"]
        logging.info(f"Model: {self.model_name}")
        # model = modelLoader(model=self.model_name, **params).to(self.device)
        model = SuperPointNet(
            encoder={}, detector_head={}, descriptor_head={}, has_dustbin=True, **params
        ).to(self.device)
        optimizer = self.get_optimizer(model, lr=self.config["model"]["learning_rate"])

        n_iter = 0
        ## new model or load pretrained
        if self.config["retrain"] == True:
            logging.info("New model")
        else:
            path = self.config["pretrained"]
            mode = (
                "" if path[-4:] == ".pth" else "full"
            )  # the suffix is '.pth' or 'tar.gz'
            logging.info("load pretrained model from: %s", path)
            model, optimizer, n_iter = pretrained_loader(
                model, optimizer, n_iter, path, mode=mode, full_path=True
            )
            logging.info("successfully load pretrained model from: %s", path)

        def set_iter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.model = model
        self.optimizer = optimizer
        self.n_iter = set_iter(n_iter)

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def validate(self):
        val_epoch_loss = 0.0
        for sample_val in self.val_loader:
            loss = self.training_step(sample_val, self.n_iter, mode="val")
            val_epoch_loss += loss

        logging.info(f"Val loss = {val_epoch_loss / len(self.val_loader) :.4f}")
        print(f"Val loss = {val_epoch_loss / len(self.val_loader) :.4f}")

    def train(self):
        logging.info(
            f"Start training: max_epoch = {self.max_epoch} | max_iter = {self.max_iter}"
        )
        losses = dict(train=[], val=[])
        epoch = 0
        for epoch in range(self.max_epoch):
            print("epoch: ", epoch)
            epoch += 1
            train_epoch_loss = 0.0
            # Validation
            for sample_train in tqdm(
                self.train_loader, desc=f"Training [{epoch}/{self.max_epoch}]"
            ):
                self.n_iter += 1
                loss, metrics = self.training_step(
                    sample_train, self.n_iter, mode="train"
                )
                losses["train"].append(loss)
                train_epoch_loss += loss
                if self.n_iter > self.max_iter:
                    logging.info("End training: %d", self.n_iter)
                    break
            msg = f"[{epoch}]Train loss = {train_epoch_loss / len(self.train_loader) :.4f}"
            msg += f"\n Metrics: {metrics}"
            logging.info(msg)
            print(msg)

            # Validation
            if self._eval:
                val_epoch_loss = 0.0
                for sample_val in self.val_loader:
                    loss, metrics = self.training_step(
                        sample_val, self.n_iter, mode="val"
                    )
                    losses["val"].append(loss)
                    val_epoch_loss += loss
                logging.info(f"Val loss = {val_epoch_loss / len(self.val_loader) :.4f}")

            # Save model
            if epoch % self.config["save_interval"] == 0:
                logging.info("Save model at epoch: %d", epoch)
                self.save_model(epoch)

    # def get_labels(
    #     self, labels_2D: torch.Tensor, cell_size: int, device: str = "cpu"
    # ) -> torch.Tensor:
    #     """
    #     # transform 2D labels to 3D shape for training
    #     :param labels_2D:
    #     :param cell_size:
    #     :param device:
    #     :return:
    #     """
    #     labels3D_flattened = labels2Dto3D_flattened(
    #         labels_2D.to(device), cell_size=cell_size
    #     )
    #     labels3D_in_loss = labels3D_flattened
    #     return labels3D_in_loss

    def get_masks(
        self, mask_2D: torch.Tensor, cell_size: int, device: str = "cpu"
    ) -> torch.Tensor:
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = reshape_pixels2superpixels(mask_2D, superpixel_size=cell_size)
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened.to(device)

    def detector_loss(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
        loss_type: str = "softmax",
    ) -> torch.Tensor:
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(preds, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction="none").cuda()
            loss = loss_func_BCE(nn.functional.softmax(preds, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def training_step(
        self,
        sample: Dict[str, torch.Tensor],
        n_iter: int = 0,
        mode: TrainerMode = "train",
    ):

        tb_interval = self.config["tensorboard_interval"]
        if_warp = self.config["data"]["warped_pair"]["enable"]

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## Get the inputs
        img, labels_2D, mask_2D, labels_3D, mask_3D_flattened = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
            sample["labels_3D"],
            sample["mask_3D_flattened"],
        )
        # Variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        labels_warp_2D = None
        img_warp = None

        # zero the parameter gradients
        self.optimizer.zero_grad()

        semi_warp = None
        # Forward + Backward + Optimize
        if mode == "train":
            outs = self.model(img.to(self.device))
            semi, coarse_desc = outs["semi"], outs["desc"]

        elif mode == "val" or mode == "test":
            with torch.no_grad():
                outs = self.model(img.to(self.device))
                semi, coarse_desc = outs["semi"], outs["desc"]

        loss = self.detector_loss(
            preds=outs["semi"],
            target=labels_3D.to(self.device),
            mask=mask_3D_flattened.to(self.device),
            loss_type=det_loss_type,
        )

        self.loss = loss

        if mode == "train":
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            precision, recall = self.compute_metrics(
                semi=semi, labels=sample["labels_2D"]
            )
        # self.scalar_dict.update(dict(loss=loss))

        # self.input_to_img_dict(sample, self.images_dict)
        # if n_iter % tb_interval == 0 or mode == "val":
        #     logging.info("Current iteration: %d", n_iter)

        #     self.log_info(
        #         semi,
        #         det_loss_type,
        #         if_warp,
        #         semi_warp,
        #         labels_2D,
        #         img,
        #         labels_warp_2D,
        #         img_warp,
        #         sample,
        #         mode,
        #     )

        # self.tb_scalar_dict(self.scalar_dict, mode)

        return loss.item(), dict(precision=precision, recall=recall)

    def compute_metrics(self, semi: torch.Tensor, labels: torch.Tensor):
        heatmap_org = self.model.detector_head.compute_heatmap(semi)
        heatmap_org_nms_batch = self.heatmap_to_nms(heatmap_org)
        precision, recall = self.batch_precision_recall(
            to_floatTensor(heatmap_org_nms_batch[:, np.newaxis, ...]),
            labels,
        )
        return precision, recall

    def log_info(
        self,
        semi: torch.Tensor,
        det_loss_type: str,
        if_warp: bool,
        semi_warp: torch.Tensor,
        labels_2D: torch.Tensor,
        img: torch.Tensor,
        labels_warp_2D: torch.Tensor,
        img_warp: torch.Tensor,
        sample: Dict[str, torch.Tensor],
        mode: TrainerMode,
    ):

        # heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
        heatmap_org = self.model.detector_head.compute_heatmap(semi)
        heatmap_org_nms_batch = self.heatmap_to_nms(heatmap_org)
        self.images_dict[f"heatmap_org_nms_batch"] = heatmap_org_nms_batch
        # if if_warp:
        #     heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
        #     heatmap_warp_nms_batch = self.heatmap_to_nms(
        #         self.images_dict, heatmap_warp, name="heatmap_warp"
        #     )

        update_overlap(
            self.images_dict,
            labels_2D,
            heatmap_org_nms_batch[None, ...],
            img,
            "original",
        )

        update_overlap(
            self.images_dict,
            labels_2D,
            get_ndarray(heatmap_org),
            img,
            "original_heatmap",
        )
        # if if_warp:
        #     update_overlap(
        #         self.images_dict,
        #         labels_warp_2D,
        #         heatmap_warp_nms_batch[np.newaxis, ...],
        #         img_warp,
        #         "warped",
        #     )
        #     update_overlap(
        #         self.images_dict,
        #         labels_warp_2D,
        #         get_ndarray(heatmap_warp),
        #         img_warp,
        #         "warped_heatmap",
        #     )

        if self.gaussian:
            # original: gt
            self.get_residual_loss(
                sample["labels_2D"],
                sample["labels_2D_gaussian"],
                sample["labels_res"],
                name="original_gt",
            )
            # if if_warp:
            #     # warped: gt
            #     self.get_residual_loss(
            #         sample["warped_labels"],
            #         sample["warped_labels_gaussian"],
            #         sample["warped_res"],
            #         name="warped_gt",
            #     )

        precision, recall = self.batch_precision_recall(
            to_floatTensor(heatmap_org_nms_batch[:, np.newaxis, ...]),
            sample["labels_2D"],
        )
        logging.info("PR_mean")
        self.scalar_dict.update({"precision": precision, "recall": recall})

        self.print_losses(self.scalar_dict, mode)
        # self.tb_images_dict(mode, self.images_dict, max_img=2)
        # self.tb_hist_dict(mode, self.hist_dict)

    def save_model(self, epoch: int = 0):
        model_state_dict = self.model.module.state_dict()
        net_state = {
            "n_iter": self.n_iter + 1,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
        }
        filename = f"{self.model_name}_e{epoch}_checkpoint.pth.tar"
        logging.info(f"Saving checkpoint to {filename}")
        torch.save(net_state, self.save_path / filename)

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter)

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                # print(f"element: {element}")
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter,
                )

    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(
                task + "-" + element, tb_dict[element], self.n_iter
            )

    def print_losses(self, losses: Dict[str, Any], task: str = "training"):
        msg = ""
        for k, v in losses.items():
            msg += f"{task} - {k}: {v.item()}\n"
        logging.info(msg)

    # def get_heatmap(
    #     self, semi: torch.Tensor, det_loss_type: str = "softmax"
    # ) -> torch.Tensor:
    #     if det_loss_type == "l2":
    #         heatmap = self.flatten_64to1(semi)
    #     else:
    #         heatmap = flattenDetection(semi)

    #     return heatmap

    @staticmethod
    def input_to_img_dict(
        sample: Dict[str, torch.Tensor], tb_images_dict: Dict[str, Any]
    ):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                tb_images_dict[k] = v
        return tb_images_dict

    @staticmethod
    def interpolate_to_dense(coarse_desc, cell_size=8):
        dense_desc = nn.functional.interpolate(
            coarse_desc, scale_factor=(cell_size, cell_size), mode="bilinear"
        )

        # norm the descriptor
        def norm_desc(desc):
            dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
            desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            return desc

        dense_desc = norm_desc(dense_desc)
        return dense_desc

    def get_residual_loss(self, labels_2D, heatmap, labels_res, name=""):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self.device
        )
        self.hist_dict[name + "_resi_loss_x"] = outs_res["loss"][:, 0]
        self.hist_dict[name + "_resi_loss_y"] = outs_res["loss"][:, 1]
        err = abs(outs_res["loss"]).mean(dim=0)
        # print("err[0]: ", err[0])
        var = abs(outs_res["loss"]).std(dim=0)
        self.scalar_dict[name + "_resi_loss_x"] = err[0]
        self.scalar_dict[name + "_resi_loss_y"] = err[1]
        self.scalar_dict[name + "_resi_var_x"] = var[0]
        self.scalar_dict[name + "_resi_var_y"] = var[1]
        self.images_dict[name + "_patches"] = outs_res["patches"]
        return outs_res

    def heatmap_to_nms(self, heatmap: torch.Tensor) -> np.ndarray:
        """
        return:
            heatmap_nms_batch: np [batch, H, W]
        """
        heatmap_np = get_ndarray(heatmap)
        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    @staticmethod
    def batch_precision_recall(
        batch_pred: torch.Tensor, batch_labels: torch.Tensor, mode: str = "sum"
    ):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precision_recall_metrics(batch_pred[i], batch_labels[i])
            precision_recall_list.append(precision_recall)
        if mode == "sum":
            precision = np.sum(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_list
                ]
            )
            recall = np.sum(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_list
                ]
            )
        elif mode == "mean":
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_list
                ]
            )
        return precision, recall

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5, device="cuda"):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        from utils.losses import norm_patches

        outs = {}
        # extract patches
        from utils.losses import extract_patches, soft_argmax_2d

        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(device), heatmap.to(device), patch_size=patch_size
        )
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        from utils.losses import do_log

        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(
            patches_log, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # extract residual
        def ext_from_points(labels_res, points):
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

        points_res = ext_from_points(labels_res, label_idx)

        # loss
        outs["pred"] = dxdy
        outs["points_res"] = points_res
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        # outs['loss'] = dxdy.cpu() - points_res.cpu()
        outs["loss"] = dxdy.to(device) - points_res.to(device)
        outs["patches"] = patches
        return outs

    @staticmethod
    def flatten_64to1(semi: torch.Tensor, cell_size: int = 8) -> torch.Tensor:
        """
        input:
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        from utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    @staticmethod
    def heatmap_nms(
        heatmap: np.ndarray, nms_dist: int = 4, conf_thresh: float = 0.015
    ) -> torch.Tensor:
        heatmap = heatmap.squeeze()
        pts_nms = get_point_from_heatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1
        return semi_thd_nms_sample
