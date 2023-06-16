import sys
import logging
import cv2
import pickle as pkl
import numpy as np
import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

sys.path.append(f'{os.getcwd()}/ControlNet')

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

def file_path(string):  # Should be moved to utils eventually
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

class PascalDataset(Dataset):
    def __init__(self, size=(512, 512)):
        self.DATASET_PATH = '/mnt/PascalVOC/VOC2012'
        
        self.image_names = list(map(lambda x: x.split('.png')[0], os.listdir(f"{self.DATASET_PATH}/SegmentationClass")))
        self.size = size
        self.default_prompt = "a high-quality, detailed, and professional image"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        target_path = f"{self.DATASET_PATH}/JPEGImages/{file_name}.jpg"
        source_path = f"{self.DATASET_PATH}/SegmentationClass/{file_name}.png"

        target = cv2.imread(target_path)[:,:,::-1]
        source = cv2.imread(source_path)[:,:,::-1]

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize
        source = cv2.resize(source, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, dsize=self.size, interpolation=cv2.INTER_CUBIC)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=self.default_prompt, hint=source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ControlNetTrain_ADE20K',
        description='Trains a ControlNet on the ADE20K dataset'
    )
    parser.add_argument('--resume-path', type=file_path)
    args = parser.parse_args()

    dataset = PascalDataset()

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    print(f"Train dataset has {len(train_ds)} images")
    print(f"Val dataset has {len(val_ds)} images")
    assert len(train_ds) + len(val_ds) == 2913

    # Configs
    resume_path = args.resume_path
    model_path = './ControlNet/models/control_sd15_ini.ckpt'
    batch_size = 8
    gpus = [2]
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
    if resume_path is None:
        model.load_state_dict(load_state_dict(model_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    train_dataloader = DataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, num_workers=0, batch_size=batch_size, shuffle=False)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(
        precision=32, 
        callbacks=[logger],
        accumulate_grad_batches=2,
        max_epochs=200,
        check_val_every_n_epoch=1, 
        default_root_dir=f'{os.getcwd()}/logs/PascalSegmentation',
        logger=True,
        accelerator='gpu',
        strategy='ddp',
        devices=gpus, 
    )


    # Train!
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_path)
