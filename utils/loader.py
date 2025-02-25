"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# from trainer.preprocessing import FromPixel2Superpixel

from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
# from settings import EXPER_PATH

from models.ops.tensor_transforms import (
    reshape_pixels2superpixels,
    reshape_superpixels2pixels,
)
from datasets.synth_dataset.base import SyntheticDataset_gaussian


class FromPixel2Superpixel:
    def __init__(
        self, superpixel_size: int, gaussian: bool = False, has_dustbin: bool = True
    ):
        self.superpixel_size = superpixel_size
        self.gaussian = gaussian
        self.has_dustbin = has_dustbin

    def space2depth(self, x: torch.Tensor) -> torch.Tensor:
        superpixel_tensor = reshape_pixels2superpixels(
            x, superpixel_size=self._superpixel_size
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

    def __call__(self, sample: Dict[str, Any]):
        if self.gaussian:
            labels_2D = sample["labels_2D_gaussian"]
        else:
            labels_2D = sample["labels_2D"]

        mask_2D = sample["valid_mask"]

        labels_3D = self.space2depth(labels_2D).float()
        mask_3D_flattened = self.get_masks(mask_2D)
        sample.update({"labels_3D": labels_3D, "mask_3D_flattened": mask_3D_flattened})
        return sample


# from utils.loader import get_save_path
def get_save_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def data_loader(
    config: dict,
    dataset: str = "syn",
    warp_input: bool = False,
    train: bool = True,
    val: bool = True,
):
    # from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset

    training_params = config.get("training", {})
    workers_train = training_params.get("workers_train", 1)  # 16
    workers_val = training_params.get("workers_val", 1)  # 16

    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }
    # if dataset == 'syn':
    #     from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
    # else:
    # Dataset = get_module("datasets", dataset)

    logging.info(f"Dataset: {dataset}")

    train_set = SyntheticDataset_gaussian(
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
    val_set = SyntheticDataset_gaussian(
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

def dataLoader(config, dataset='syn', warp_input=False, train=True, val=True):
    import torchvision.transforms as transforms
    training_params = config.get('training', {})
    workers_train = training_params.get('workers_train', 1) # 16
    workers_val   = training_params.get('workers_val', 1) # 16
        
    logging.info(f"workers_train: {workers_train}, workers_val: {workers_val}")
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    # if dataset == 'syn':
    #     from datasets.SyntheticDataset_gaussian import SyntheticDataset as Dataset
    # else:
    Dataset = get_module('datasets', dataset)
    print(f"dataset: {dataset}")

    train_set = Dataset(
        transform=data_transforms['train'],
        task = 'train',
        **config['data'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['model']['batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn
    )
    val_set = Dataset(
        transform=data_transforms['train'],
        task = 'val',
        **config['data'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn
    )
    # val_set, val_loader = None, None
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}

def dataLoader_test(config, dataset='syn', warp_input=False, export_task='train'):
    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 1) # 16
    logging.info(f"workers_test: {workers_test}")

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }
    test_loader = None
    if dataset == 'syn':
        from datasets.SyntheticDataset import SyntheticDataset
        test_set = SyntheticDataset(
            transform=data_transforms['test'],
            train=False,
            warp_input=warp_input,
            getPts=True,
            seed=1,
            **config['data'],
        )
    elif dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    # elif dataset == 'Coco' or 'Kitti' or 'Tum':
    else:
        # from datasets.Kitti import Kitti
        logging.info(f"load dataset from : {dataset}")
        Dataset = get_module('datasets', dataset)
        test_set = Dataset(
            export=True,
            task=export_task,
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn

        )
    return {'test_set': test_set, 'test_loader': test_loader}

def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)

def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)

def modelLoader(model='SuperPointNet', **kwargs):
    # create model
    logging.info("=> creating model: %s", model)
    if model == "SuperPointNet":
        from models.superpoint.superpoint_net import SuperPointNet
        net = SuperPointNet(encoder={}, detector_head={}, descriptor_head={}, has_dustbin=True, **kwargs)
    else:
        net = get_model(model)
        net = net(**kwargs)
    return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    if mode == 'full':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint['epoch']
        epoch = checkpoint['n_iter']
#         epoch = 0
    else:
        net.load_state_dict(checkpoint)
        # net.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage))
    return net, optimizer, epoch

if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')
