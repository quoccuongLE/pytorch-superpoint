import logging
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.utils.data as data
from imageio import imread
from numpy.linalg import inv
from tqdm import tqdm

from datasets import synthetic_dataset
from datasets.data_tools import get_labels_bi, np_to_tensor
from datasets.data_tools import warpLabels as warp_labels
from datasets.ops.preprocessing import FromPixel2Superpixel
from settings import DATA_PATH
from settings import DEBUG as debug
from settings import SYN_TMPDIR
from utils.homographies import sample_homography_np as sample_homography
from utils.photometric import ImgAugTransform as img_aug_transform
from utils.photometric import customizedTransform as customized_transform
from utils.tools import dict_update
from utils.utils import compute_valid_mask, filter_points
from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import inv_warp_image, is_tar_extracted, warp_points
from utils.var_dim import squeezeToNumpy as squeeze2np

TMPDIR = SYN_TMPDIR  # './datasets/' # you can define your tmp dir


def _load_img_as_float(path: str) -> np.ndarray:
    return imread(path).astype(np.float32) / 255


def _parse_primitives(
    names: Union[str, List[str]], all_primitives: List[str]
) -> List[str]:
    if isinstance(names, str):
        primitives = all_primitives if names == "all" else [names]
    elif isinstance(names, list):
        primitives = names
    assert set(primitives) <= set(all_primitives)
    return primitives


def _get_labels(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    labels = torch.zeros(height, width)
    pnts_int = torch.min(
        points.round().long(), torch.tensor([[width - 1, height - 1]]).long()
    )
    labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
    return labels


def _get_label_res(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    quan = lambda x: x.round().long()
    labels_res = torch.zeros(height, width, 2)
    labels_res[quan(points)[:, 1], quan(points)[:, 0], :] = points - points.round()
    labels_res = labels_res.transpose(1, 2).transpose(0, 1)
    return labels_res


class SyntheticDataset_gaussian(data.Dataset):

    default_config = {
        "primitives": "all",
        "truncate": {},
        "validation_size": -1,
        "test_size": -1,
        "on-the-fly": False,
        "cache_in_memory": False,
        "suffix": None,
        "add_augmentation_to_test_set": False,
        "num_parallel_calls": 10,
        "generation": {
            "split_sizes": {"training": 10000, "validation": 200, "test": 500},
            "image_size": [960, 1280],
            "random_seed": 0,
            "params": {
                "generate_background": {
                    "min_kernel_size": 150,
                    "max_kernel_size": 500,
                    "min_rad_ratio": 0.02,
                    "max_rad_ratio": 0.031,
                },
                "draw_stripes": {"transform_params": (0.1, 0.1)},
                "draw_multiple_polygons": {"kernel_boundaries": (50, 100)},
            },
        },
        "preprocessing": {
            "resize": [240, 320],
            "blur_size": 11,
        },
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {
                "enable": False,
                "params": {},
                "valid_border_margin": 0,
            },
        },
    }

    # debug = True

    if debug == True:
        drawing_primitives = [
            "draw_checkerboard",
        ]
    else:
        drawing_primitives = [
            "draw_lines",
            "draw_polygon",
            "draw_multiple_polygons",
            "draw_ellipses",
            "draw_star",
            "draw_checkerboard",
            "draw_stripes",
            "draw_cube",
            "gaussian_noise",
        ]
    logging.info(drawing_primitives)

    def dump_primitive_data(self, primitive: str, tar_path: str, config: dict):
        temp_dir = Path(TMPDIR, primitive)
        logging.info("Generating tarfile for primitive {}.".format(primitive))
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config["generation"]["image_size"],
                    **config["generation"]["params"]["generate_background"],
                )
                points = np.array(
                    getattr(synthetic_dataset, primitive)(
                        image, **config["generation"]["params"].get(primitive, {})
                    )
                )
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                    points
                    * np.array(config["preprocessing"]["resize"], np.float32)
                    / np.array(config["generation"]["image_size"], np.float32)
                )
                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)

        # Pack into a tar file
        with tarfile.open(tar_path, mode="w:gz") as tar:
            tar.add(temp_dir, arcname=primitive)
        shutil.rmtree(temp_dir)
        logging.info("Tarfile dumped to {}.".format(tar_path))

    def __init__(
        self,
        seed: int = None,
        task: str = "train",
        transform: callable = None,
        get_points: bool = False,
        primitives: str = "draw_checkerboard",
        suffix: str = None,
        to_check: bool = False,
        **kwargs,
    ):

        self.from_pixel2superpixel = FromPixel2Superpixel(superpixel_size=8)

        torch.set_default_dtype(torch.float32)
        np.random.seed(seed)
        random.seed(seed)

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, dict(kwargs))

        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.img_aug_transform = img_aug_transform
        self.customized_transform = customized_transform

        ######
        self.enable_photo_train = self.config["augmentation"]["photometric"]["enable"]
        self.enable_homo_train = self.config["augmentation"]["homographic"]["enable"]
        self.enable_homo_val = False
        self.enable_photo_val = False
        ######

        self.action = "training" if task == "train" else "validation"

        self.cell_size = 8
        self.get_points = get_points

        self.gaussian_label = self.config["gaussian_label"].get("enable", False)

        # Parse drawing primitives
        primitive_list = _parse_primitives(primitives, self.drawing_primitives)

        basepath = Path(
            DATA_PATH,
            "synthetic_shapes" + ("_{}".format(suffix) if suffix is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {"images": [], "points": []} for s in [self.action]}
        for primitive in primitive_list:
            tar_path = Path(basepath, "{}.tar.gz".format(primitive))
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, self.config)

            # Untar locally
            temp_dir = Path(os.environ.get("TMPDIR", TMPDIR))
            if to_check and not is_tar_extracted(
                tar_filepath=tar_path, extraction_directory=temp_dir
            ):
                logging.info("Extracting archive for primitive {}.".format(primitive))
                logging.info(f"tar_path: {tar_path}")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=temp_dir)
            else:
                logging.info(f"Using exising data folder for primitive {primitive}")

            # Gather filenames in all splits, optionally truncate
            truncate = self.config["truncate"].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, "images", s).iterdir()]
                f = [p.replace("images", "points") for p in e]
                f = [p.replace(".png", ".npy") for p in f]
                splits[s]["images"].extend(e[: int(truncate * len(e))])
                splits[s]["points"].extend(f[: int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]["images"]))
            for obj in ["images", "points"]:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)

    def crawl_folders(self, splits: dict):
        sequence_set = []
        for img, pnts in zip(
            splits[self.action]["images"], splits[self.action]["points"]
        ):
            sample = {"image": img, "points": pnts}
            sequence_set.append(sample)
        self.samples = sequence_set

    def put_gaussian_maps(
        self, center: np.ndarray, accumulate_confid_map: np.ndarray
    ) -> np.ndarray:
        crop_size_y = self.params_transform["crop_size_y"]
        crop_size_x = self.params_transform["crop_size_x"]
        stride = self.params_transform["stride"]
        sigma = self.params_transform["sigma"]

        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)), indexing="xy")
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= sigma
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
        return accumulate_confid_map

    def img_photometric(self, img: np.ndarray) -> np.ndarray:
        augmentation = self.img_aug_transform(**self.config["augmentation"])
        new_img = augmentation(img[:, :, np.newaxis])
        cusAug = self.customized_transform()
        new_img = cusAug(new_img, **self.config["augmentation"])
        return new_img

    def img_homographic(self, img: np.ndarray, points: np.ndarray):
        homography = self.sample_homography(
            np.array([2, 2]),
            shift=-1,
            **self.config["augmentation"]["homographic"]["params"],
        )

        homography = inv(homography)
        homography = torch.tensor(homography).float()
        inv_homography = homography.inverse()
        img = torch.from_numpy(img)
        warped_img = self.inv_warp_image(img.squeeze(), inv_homography, mode="bilinear")
        warped_img = warped_img.squeeze().numpy()
        warped_img = warped_img[:, :, np.newaxis]

        warped_points = self.warp_points(
            points, homography_scaling(homography, self.H, self.W)
        )
        warped_points = filter_points(warped_points, torch.tensor([self.W, self.H]))

        if self.transform:
            warped_img = self.transform(warped_img)

        valid_mask = self.compute_valid_mask(
            torch.tensor([self.H, self.W]),
            inv_homography=inv_homography,
            erosion_radius=self.config["augmentation"]["homographic"][
                "valid_border_margin"
            ],
        )  # can set to other value
        labels_2D = _get_labels(warped_points, self.H, self.W).unsqueeze(0)
        labels_res = _get_label_res(warped_points, self.H, self.W)
        return warped_img, labels_2D, valid_mask, labels_res, warped_points

    def default_img_transform(self, img: np.ndarray, points: np.ndarray):
        new_img = img[:, :, np.newaxis]
        if self.transform:
            new_img = self.transform(new_img)
        valid_mask = self.compute_valid_mask(
            torch.tensor([self.H, self.W]), inv_homography=torch.eye(3)
        )
        labels_res = _get_label_res(points, self.H, self.W)
        return new_img, labels_res, valid_mask

    def __getitem__(self, index: int):

        sample = self.samples[index]
        img = _load_img_as_float(sample["image"])
        H, W = img.shape[0], img.shape[1]
        self.H = H
        self.W = W
        gt_points = np.load(sample["points"])  # (y, x)
        gt_points = torch.tensor(gt_points).float()
        gt_points = torch.stack((gt_points[:, 1], gt_points[:, 0]), dim=1)  # (x, y)
        gt_points = filter_points(gt_points, torch.tensor([W, H]))
        sample = {}

        labels_2D = _get_labels(gt_points, H, W)
        sample["labels_2D"] = labels_2D.unsqueeze(0)

        if (
            self.config["augmentation"]["photometric"]["enable_train"]
            and self.action == "training"
        ) or (
            self.config["augmentation"]["photometric"]["enable_val"]
            and self.action == "validation"
        ):
            img = self.img_photometric(img)

        if (
            self.config["augmentation"]["homographic"]["enable_train"]
            and self.action == "training"
        ) or (
            self.config["augmentation"]["homographic"]["enable_val"]
            and self.action == "validation"
        ):
            warped_img, labels_2D, valid_mask, labels_res, pnts_post = (
                self.img_homographic(img=img, points=gt_points)
            )
            sample["image"] = warped_img
            sample["labels_2D"] = labels_2D
            sample["valid_mask"] = valid_mask
            sample["labels_res"] = labels_res
        else:
            new_img, labels_res, valid_mask = self.default_img_transform(
                img=img, points=gt_points
            )
            pnts_post = gt_points
            sample["valid_mask"] = valid_mask
            sample["image"] = new_img
            sample["labels_res"] = labels_res

        if self.gaussian_label:
            labels_2D_bi = get_labels_bi(pnts_post, H, W)
            labels_gaussian = self.gaussian_blur(squeeze2np(labels_2D_bi))
            labels_gaussian = np_to_tensor(labels_gaussian, H, W)
            sample["labels_2D_gaussian"] = labels_gaussian

        if self.config["warped_pair"]["enable"]:
            (
                homography,
                inv_homography,
                warped_img,
                warped_labels,
                warped_res,
                warped_labels_gaussian,
                warped_labels_bi,
                valid_mask,
            ) = self.augment_pairing_sample(img=img, points=gt_points)
            sample["warped_img"] = warped_img
            sample["warped_labels"] = warped_labels
            sample["warped_res"] = warped_res
            sample["warped_valid_mask"] = valid_mask
            sample["homographies"] = homography
            sample["inv_homographies"] = inv_homography
            if self.gaussian_label:
                sample["warped_labels_gaussian"] = warped_labels_gaussian
                sample["warped_labels_bi"] = warped_labels_bi

        if self.get_points:
            sample["gts"] = gt_points

        labels_3D = self.from_pixel2superpixel.space2depth(
            sample["labels_2D"][None, ...]
        ).float()
        mask_3D_flattened = self.from_pixel2superpixel.get_masks(
            sample["valid_mask"][None, ...]
        )
        sample["labels_3D"] = labels_3D[0]
        sample["mask_3D_flattened"] = mask_3D_flattened[0]
        return sample

    def augment_pairing_sample(self, img: torch.Tensor, points: np.ndarray):
        homography = self.sample_homography(
            np.array([2, 2]), shift=-1, **self.config["warped_pair"]["params"]
        )

        # Use inverse from the sample homography
        homography = np.linalg.inv(homography)
        inv_homography = np.linalg.inv(homography)

        homography = torch.tensor(homography).type(torch.float32)
        inv_homography = torch.tensor(inv_homography).type(torch.float32)

        # Photometric augmentation from original image
        warped_img = img.type(torch.float32)
        warped_img = self.inv_warp_image(
            warped_img.squeeze(), inv_homography, mode="bilinear"
        ).unsqueeze(0)
        if (self.enable_photo_train == True and self.action == "train") or (
            self.enable_photo_val and self.action == "validation"
        ):
            warped_img = self.img_photometric(
                warped_img.numpy().squeeze()
            )  # Numpy array (H, W, 1)
            warped_img = torch.tensor(warped_img, dtype=torch.float32)

        warped_img = warped_img.view(-1, self.H, self.W)

        warped_set = warp_labels(points, self.H, self.W, homography, bilinear=True)
        warped_labels = warped_set["labels"]
        warped_res = warped_set["res"].transpose(1, 2).transpose(0, 1)

        if self.gaussian_label:
            warped_labels_bi = warped_set["labels_bi"]
            warped_labels_gaussian = self.gaussian_blur(
                squeeze2np(warped_labels_bi)
            )
            warped_labels_gaussian = np_to_tensor(
                warped_labels_gaussian, self.H, self.W
            )
        else:
            warped_labels_gaussian = None
            warped_labels_bi = None

        valid_mask = self.compute_valid_mask(
            torch.tensor([self.H, self.W]),
            inv_homography=inv_homography,
            erosion_radius=self.config["warped_pair"]["valid_border_margin"],
        )  # can set to other value
        return (
            homography,
            inv_homography,
            warped_img,
            warped_labels,
            warped_res,
            warped_labels_gaussian,
            warped_labels_bi,
            valid_mask,
        )

    def __len__(self):
        return len(self.samples)

    ## util functions
    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: np [H, W]
        Returns:
            blurred_image: np [H, W]
        """
        aug_par = {"photometric": {}}
        aug_par["photometric"]["enable"] = True
        aug_par["photometric"]["params"] = self.config["gaussian_label"]["params"]
        augmentation = self.img_aug_transform(**aug_par)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
