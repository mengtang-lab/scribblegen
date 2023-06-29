from typing import Tuple, Dict

import cv2
import pickle as pkl
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from utils import loadAde20K
from config import DatasetEnum, ExpConfig

class ADE20KDataset(Dataset):
    def __init__(self, size: Tuple[int, int] = (512, 512), train: bool = True, overfit: bool = False):
        self.DATASET_PATH = '/mnt/ADE20K'
        index_file = 'ADE20K_2021_17_01/index_ade20k.pkl'
        with open('{}/{}'.format(self.DATASET_PATH, index_file), 'rb') as f:
            self.index_ade20k = pkl.load(f)
        split = 'train' if train else 'val'
        self.indices = [
            i for i in range(len(self.index_ade20k['filename'])) 
            if split in self.index_ade20k['filename'][i]
        ]
        if overfit:
            self.indices = self.indices[:10]
        
        self.size = size
        self.default_prompt = "a high-quality, detailed, and professional image"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        # Make idx go from index in the current dataset (i.e. restricted to train or val) to the
        #   true index in the total dataset of ADE20K.
        idx = self.indices[idx]

        full_file_name = '{}/{}'.format(self.index_ade20k['folder'][idx], self.index_ade20k['filename'][idx])
        info = loadAde20K('{}/{}'.format(self.DATASET_PATH, full_file_name))
        target = cv2.imread(info['img_name'])[:,:,::-1]
        source = cv2.imread(info['segm_name'])[:,:,::-1]

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


class PascalSegmentationDataset(Dataset):
    def __init__(self, size: Tuple[int, int] = (512, 512), train: bool = True, overfit: bool = False):
        self.DATASET_PATH = '/mnt/PascalVOC'
        
        with open(f"{self.DATASET_PATH}/{'train.txt' if train else 'val.txt'}", 'r') as f:
            self.image_names = f.readlines()
        self.image_names = list(map(lambda n: n[:-1], self.image_names))
        self.size = size
        self.default_prompt = "a high-quality, detailed, and professional image"
        if overfit:
            self.image_names = self.image_names[:10]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict:
        file_name = self.image_names[idx]
        target_path = f"{self.DATASET_PATH}/JPEGImages/{file_name}.jpg"
        source_path = f"{self.DATASET_PATH}/SegmentationClassAug/{file_name}.png"

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


def get_dataloaders(config: ExpConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds: Dataset
    val_ds: Dataset
    if config.dataset.value == DatasetEnum.ADE20K.value:
        train_ds = ADE20KDataset(size=tuple(config.image_size), train=True, overfit=config.overfit)
        val_ds = ADE20KDataset(size=tuple(config.image_size), train=False, overfit=config.overfit)
        assert len(train_ds) + len(val_ds) == 27258 or config.overfit

    elif config.dataset.value == DatasetEnum.PascalSegmentation.value:
        train_ds = PascalSegmentationDataset(size=tuple(config.image_size), train=True, overfit=config.overfit)
        val_ds = PascalSegmentationDataset(size=tuple(config.image_size), train=False, overfit=config.overfit)
        assert len(train_ds) + len(val_ds) == 12031 or config.overfit

    else:
        raise NotImplementedError
    
    print(f"Train dataset has {len(train_ds)} images")
    print(f"Val dataset has {len(val_ds)} images")
    train_dataloader = DataLoader(train_ds, num_workers=config.num_workers, 
                                  batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, num_workers=config.num_workers, 
                                batch_size=config.batch_size, shuffle=False)
    return train_dataloader, val_dataloader
