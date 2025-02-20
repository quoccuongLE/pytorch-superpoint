import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

import torch.optim
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from utils.loader import modelLoader, pretrainedLoader
from utils.tools import dict_update
from utils.utils import (
    flattenDetection,
    labels2Dto3D,
    labels2Dto3D_flattened,
    precisionRecall_torch,
    save_checkpoint,
)

from .utils import get_ndarray, img_overlap, update_overlap, to_floatTensor


TrainerMode = Literal["train", "val", "test"]


class Trainer:
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

        self.max_iter = config["train_iter"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss

            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        self.print_important_config()

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
        self.net = nn.DataParallel(self.net)
        self.optimizer = self.get_optimizer(
            self.net, lr=self.config["model"]["learning_rate"]
        )

    def get_optimizer(self, net: nn.Module, lr: float):
        """
        initiate adam optimizer
        :param net: network structure
        :param lr: learning rate
        :return:
        """
        print("ADAM optimizer")
        return optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    def load_model(self):
        """
        load model from name and params
        init or load optimizer
        :return:
        """
        model = self.config["model"]["name"]
        params = self.config["model"]["params"]
        print("model: ", model)
        net = modelLoader(model=model, **params).to(self.device)
        logging.info("=> setting adam solver")
        optimizer = self.get_optimizer(net, lr=self.config["model"]["learning_rate"])

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
            net, optimizer, n_iter = pretrainedLoader(
                net, optimizer, n_iter, path, mode=mode, full_path=True
            )
            logging.info("successfully load pretrained model from: %s", path)

        def set_iter(n_iter):
            if self.config["reset_iter"]:
                logging.info("reset iterations to 0")
                n_iter = 0
            return n_iter

        self.net = net
        self.optimizer = optimizer
        self.n_iter = set_iter(n_iter)

    @property
    def train_loader(self):
        """
        loader for dataset, set from outside
        :return:
        """
        print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    @property
    def val_loader(self):
        print("get dataloader")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        print("set train loader")
        self._val_loader = loader

    def train(self, **kwargs):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :param options:
        :return:
        """
        # training info
        logging.info("n_iter: %d", self.n_iter)
        logging.info("max_iter: %d", self.max_nb_epoch)
        running_losses = []
        epoch = 0
        # Train one epoch
        while self.n_iter < self.max_nb_epoch:
            print("epoch: ", epoch)
            epoch += 1
            for i, sample_train in tqdm(enumerate(self.train_loader)):
                # train one sample
                loss_out = self.training_step(sample_train, self.n_iter, mode="train")
                self.n_iter += 1
                running_losses.append(loss_out)
                # run validation
                if self._eval and self.n_iter % self.config["validation_interval"] == 0:
                    logging.info("====== Validating...")
                    for j, sample_val in enumerate(self.val_loader):
                        self.training_step(sample_val, self.n_iter + j, mode="val")
                        if j > self.config.get("validation_size", 3):
                            break
                # save model
                if self.n_iter % self.config["save_interval"] == 0:
                    logging.info(
                        "save model: every %d interval, current iteration: %d",
                        self.config["save_interval"],
                        self.n_iter,
                    )
                    self.save_model()
                # ending condition
                if self.n_iter > self.max_nb_epoch:
                    # end training
                    logging.info("End training: %d", self.n_iter)
                    break

    def get_labels(self, labels_2D, cell_size, device="cpu"):
        """
        # transform 2D labels to 3D shape for training
        :param labels_2D:
        :param cell_size:
        :param device:
        :return:
        """
        labels3D_flattened = labels2Dto3D_flattened(
            labels_2D.to(device), cell_size=cell_size
        )
        labels3D_in_loss = labels3D_flattened
        return labels3D_in_loss

    def get_masks(self, mask_2D, cell_size, device="cpu"):
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
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened

    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
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
            loss = loss_func(input, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction="none").cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def training_step(
        self,
        sample: dict[str, torch.Tensor],
        n_iter: int = 0,
        mode: TrainerMode = "train",
    ):

        tb_interval = self.config["tensorboard_interval"]
        if_warp = self.config["data"]["warped_pair"]["enable"]

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## get the inputs
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config["model"]["detector_loss"]["loss_type"]
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        # img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'].to(self.device), \
        #     sample['warped_labels'].to(self.device), \
        #     sample['warped_valid_mask'].to(self.device)
        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_labels"],
                sample["warped_valid_mask"],
            )

        # homographies
        # mat_H, mat_H_inv = \
        # sample['homographies'].to(self.device), sample['inv_homographies'].to(self.device)
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if mode == "train":
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs = self.net(img.to(self.device))
            semi, coarse_desc = outs["semi"], outs["desc"]
            if if_warp:
                outs_warp = self.net(img_warp.to(self.device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]

        elif mode == "val" or mode == "test":
            with torch.no_grad():
                outs = self.net(img.to(self.device))
                semi, coarse_desc = outs["semi"], outs["desc"]
                if if_warp:
                    outs_warp = self.net(img_warp.to(self.device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                pass

        # detector loss
        from utils.utils import labels2Dto3D

        if self.gaussian:
            labels_2D = sample["labels_2D_gaussian"]
            if if_warp:
                warped_labels = sample["warped_labels_gaussian"]
        else:
            labels_2D = sample["labels_2D"]
            if if_warp:
                warped_labels = sample["warped_labels"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":
            add_dustbin = True

        labels_3D = labels2Dto3D(
            labels_2D.to(self.device), cell_size=self.cell_size, add_dustbin=add_dustbin
        ).float()
        mask_3D_flattened = self.get_masks(mask_2D, self.cell_size, device=self.device)
        loss_det = self.detector_loss(
            input=outs["semi"],
            target=labels_3D.to(self.device),
            mask=mask_3D_flattened,
            loss_type=det_loss_type,
        )
        # warp
        if if_warp:
            labels_3D = labels2Dto3D(
                warped_labels.to(self.device),
                cell_size=self.cell_size,
                add_dustbin=add_dustbin,
            ).float()
            mask_3D_flattened = self.get_masks(
                mask_warp_2D, self.cell_size, device=self.device
            )
            loss_det_warp = self.detector_loss(
                input=outs_warp["semi"],
                target=labels_3D.to(self.device),
                mask=mask_3D_flattened,
                loss_type=det_loss_type,
            )
        else:
            loss_det_warp = torch.tensor([0]).float().to(self.device)

        ## get labels, masks, loss for detection
        # labels3D_in_loss = self.get_labels(labels_2D, self.cell_size, device=self.device)
        # mask_3D_flattened = self.get_masks(mask_2D, self.cell_size, device=self.device)
        # loss_det = self.get_loss(semi, labels3D_in_loss, mask_3D_flattened, device=self.device)

        ## warping
        # labels3D_in_loss = self.get_labels(labels_warp_2D, self.cell_size, device=self.device)
        # mask_3D_flattened = self.get_masks(mask_warp_2D, self.cell_size, device=self.device)
        # loss_det_warp = self.get_loss(semi_warp, labels3D_in_loss, mask_3D_flattened, device=self.device)

        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self.config["model"]["lambda_loss"]

        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                coarse_desc,
                coarse_desc_warp,
                mat_H,
                mask_valid=mask_desc,
                device=self.device,
                **self.desc_params
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        # REMOVE: add_res_loss
        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        self.input_to_img_dict(sample, self.images_dict)

        if mode == "train":
            loss.backward()
            self.optimizer.step()

        if n_iter % tb_interval == 0 or mode == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )

            self.get_stats_info(
                semi,
                det_loss_type,
                if_warp,
                semi_warp,
                labels_2D,
                img,
                labels_warp_2D,
                img_warp,
                sample,
                mode
            )

        self.tb_scalar_dict(self.scalar_dict, mode)

        return loss.item()

    def get_stats_info(
        self,
        semi: torch.Tensor,
        det_loss_type: str,
        if_warp: bool,
        semi_warp: torch.Tensor,
        labels_2D: torch.Tensor,
        img: torch.Tensor,
        labels_warp_2D: torch.Tensor,
        img_warp: torch.Tensor,
        sample: dict[str, torch.Tensor],
        mode: TrainerMode
    ):

        heatmap_org = self.get_heatmap(semi, det_loss_type)  # tensor []
        heatmap_org_nms_batch = self.heatmap_to_nms(
            self.images_dict, heatmap_org, name="heatmap_org"
        )
        if if_warp:
            heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
            heatmap_warp_nms_batch = self.heatmap_to_nms(
                self.images_dict, heatmap_warp, name="heatmap_warp"
            )

        update_overlap(
            self.images_dict,
            labels_2D,
            heatmap_org_nms_batch[np.newaxis, ...],
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
        if if_warp:
            update_overlap(
                self.images_dict,
                labels_warp_2D,
                heatmap_warp_nms_batch[np.newaxis, ...],
                img_warp,
                "warped",
            )
            update_overlap(
                self.images_dict,
                labels_warp_2D,
                get_ndarray(heatmap_warp),
                img_warp,
                "warped_heatmap",
            )

        if self.gaussian:
            # original: gt
            self.get_residual_loss(
                sample["labels_2D"],
                sample["labels_2D_gaussian"],
                sample["labels_res"],
                name="original_gt",
            )
            if if_warp:
                # warped: gt
                self.get_residual_loss(
                    sample["warped_labels"],
                    sample["warped_labels_gaussian"],
                    sample["warped_res"],
                    name="warped_gt",
                )

        pr_mean = self.batch_precision_recall(
            to_floatTensor(heatmap_org_nms_batch[:, np.newaxis, ...]),
            sample["labels_2D"],
        )
        print("pr_mean")
        self.scalar_dict.update(pr_mean)

        self.print_losses(self.scalar_dict, mode)
        self.tb_images_dict(mode, self.images_dict, max_img=2)
        self.tb_hist_dict(mode, self.hist_dict)

    def save_model(self):
        """
        # save checkpoint for resuming training
        :return:
        """
        model_state_dict = self.net.module.state_dict()
        save_checkpoint(
            self.save_path,
            {
                "n_iter": self.n_iter + 1,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.n_iter,
        )

    def add_single_image_to_tb(self, task, img_tensor, n_iter, name="img"):
        """
        # add image to tensorboard for visualization
        :param task:
        :param img_tensor:
        :param n_iter:
        :param name:
        :return:
        """
        if img_tensor.dim() == 4:
            for i in range(min(img_tensor.shape[0], 5)):
                self.writer.add_image(
                    task + "-" + name + "/%d" % i, img_tensor[i, :, :, :], n_iter
                )
        else:
            self.writer.add_image(task + "-" + name, img_tensor[:, :, :], n_iter)

    def tb_scalar_dict(self, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self.writer.add_scalar(task + "-" + element, losses[element], self.n_iter)
            # print (task, '-', element, ": ", losses[element].item())

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
        pass

    def print_losses(self, losses, task="training"):
        """
        # print loss for tracking training
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            # print ('add to tb: ', element)
            print(task, "-", element, ": ", losses[element].item())

    def add2tensorboard_nms(self, img, labels_2D, semi, task="training", batch_size=1):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        from utils.utils import box_nms, getPtsFromHeatmap

        boxNms = False
        n_iter = self.n_iter

        nms_dist = self.config["model"]["nms"]
        conf_thresh = self.config["model"]["detection_threshold"]
        # print("nms_dist: ", nms_dist)
        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = get_ndarray(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[
                pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)
            ] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
            # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
            # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    get_ndarray(img[idx, :, :, :]),
                )
                self.writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    n_iter,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(
                torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample
            )
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(
                    semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh
                ).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        get_ndarray(img[idx, :, :, :]),
                    )
                    self.writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        n_iter,
                    )
                precision_recall_boxnms = precisionRecall_torch(
                    semi_flat_tensor_nms, label_sample_nms_sample
                )
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        self.writer.add_scalar(task + "-precision_nms", precision, n_iter)
        self.writer.add_scalar(task + "-recall_nms", recall, n_iter)
        print(
            "-- [%s-%d-fast NMS] precision: %.4f, recall: %.4f"
            % (task, n_iter, precision, recall)
        )
        if boxNms:
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            self.writer.add_scalar(task + "-precision_boxnms", precision, n_iter)
            self.writer.add_scalar(task + "-recall_boxnms", recall, n_iter)
            print(
                "-- [%s-%d-boxNMS] precision: %.4f, recall: %.4f"
                % (task, n_iter, precision, recall)
            )

    def get_heatmap(self, semi, det_loss_type="softmax"):
        if det_loss_type == "l2":
            heatmap = self.flatten_64to1(semi)
        else:
            heatmap = flattenDetection(semi)
        return heatmap

    ######## static methods ########
    @staticmethod
    def input_to_img_dict(sample: dict[str, torch.Tensor], tb_images_dict: dict):
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

    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return:
            heatmap_nms_batch: np [batch, H, W]
        """
        heatmap_np = get_ndarray(heatmap)
        ## heatmap_nms
        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
        # images_dict.update({name + '_nms_batch': heatmap_nms_batch})
        images_dict.update({name + "_nms_batch": heatmap_nms_batch[:, np.newaxis, ...]})
        return heatmap_nms_batch

    @staticmethod
    def batch_precision_recall(batch_pred, batch_labels):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precisionRecall_torch(batch_pred[i], batch_labels[i])
            precision_recall_list.append(precision_recall)
        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        return {"precision": precision, "recall": recall}

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
    def flatten_64to1(semi, cell_size=8):
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
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from utils.utils import getPtsFromHeatmap

        # nms_dist = self.config['model']['nms']
        # conf_thresh = self.config['model']['detection_threshold']
        heatmap = heatmap.squeeze()
        # print("heatmap: ", heatmap.shape)
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1
        return semi_thd_nms_sample
