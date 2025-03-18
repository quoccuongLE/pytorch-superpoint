import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim
import torch.utils.data
import yaml

from settings import EXPER_PATH


def get_save_path(output_dir):
    save_path = Path(output_dir)
    save_path = save_path / "checkpoints"
    logging.info("Save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def log_dataset_batch_info(train_loader, config: Dict[str, Any], tag='train'):
    batch_size = config['model']['batch_size']
    logging.info(
        f"{tag} dataset splitted by batch size {batch_size} into {len(train_loader) // batch_size} batches"
    )


def train(config: Dict[str, Any], output_dir: str, args: Dict[str, Any]):
    # TODO: logging error
    from utils.loader import data_loader
    from trainer import BaseTrainer

    assert 'train_iter' in config

    torch.set_default_dtype(torch.float32)
    dataset_cfg = config['data']['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Train on device: %s", device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    save_path = get_save_path(output_dir)

    # Data loading
    data = data_loader(config, dataset=dataset_cfg)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    log_dataset_batch_info(train_loader, config, tag='train')
    log_dataset_batch_info(val_loader, config, tag='val')

    train_agent = BaseTrainer(config, save_path=save_path, device=device)

    # Writer from tensorboard
    train_agent.writer = None

    # Feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # Load model initiates the model and load the pretrained model (if any)
    train_agent.load_model()
    train_agent.data_parallel()

    if args.eval:
        train_agent.validate()
    else:
        try:
            train_agent.train()
        except KeyboardInterrupt:
            print ("Press Ctrl + C, saving model!")
            train_agent.save_model()


if __name__ == '__main__':
    # Global var
    torch.set_default_dtype(torch.float32)

    # Add parser
    parser = argparse.ArgumentParser()
    # Training command
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", type=str)
    parser.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    parser.add_argument("--eval", action="store_true", default=False)

    args = parser.parse_args()
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
        filename=output_dir + f"/train_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",
    )

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logging.info("Running")
    train(config, output_dir, args)
