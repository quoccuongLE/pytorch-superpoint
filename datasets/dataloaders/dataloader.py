import logging

import numpy as np
import torch
import torchvision.transforms as transforms

from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
from ..ops.preprocessing import FromPixel2Superpixel

def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


def data_loader(config: dict, dataset: str = "syn", warp_input: bool = False, train: bool = True, val: bool = True):

    training_params = config.get("training", {})
    workers_train = training_params.get("workers_train", 1)  # 16
    workers_val = training_params.get("workers_val", 1)  # 16

    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    data_transforms = {
        "train": transforms.Compose(
            [
                FromPixel2Superpixel(superpixel_size=8),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                FromPixel2Superpixel(superpixel_size=8),
                transforms.ToTensor(),
            ]
        ),
    }
    # if dataset == 'syn':
    #     from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
    # else:
    # Dataset = get_module("datasets", dataset)

    print(f"dataset: {dataset}")

    train_set = Dataset(
        transform=data_transforms["train"],
        task="train",
        **config["data"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn,
    )
    val_set = Dataset(
        transform=data_transforms["train"],
        task="val",
        **config["data"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config["model"]["eval_batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn,
    )
    # val_set, val_loader = None, None
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_set": train_set,
        "val_set": val_set,
    }
