from typing import Tuple, Dict, Optional

import cv2
import pickle as pkl
import numpy as np
import json
import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from config import DatasetEnum, ExpConfig


class PascalSegmentationDataset(Dataset):
    def __init__(
            self, 
            size: Tuple[int, int] = (512, 512), # Input image size
            train: bool = True, # Whether to use training split or not
            overfit: bool = False, # Whether to limit the dataset to 10 images
            class_hint: bool = False, # Whether to include class names in the CLIP prompt
        ):
        self.DATASET_PATH = '/path/to/datasets/PascalVOC'
        
        with open(f"{self.DATASET_PATH}/{'train.txt' if train else 'val.txt'}", 'r') as f:
            self.image_names = f.readlines()
        # Remove new line character
        self.image_names = list(map(lambda n: n[:-1], self.image_names))

        # Get present classes
        self.class_hint = class_hint
        if class_hint:
            with open(f"{self.DATASET_PATH}/classes.json", 'r') as f:
                self.classes = json.load(f)
        else:
            self.classes = None

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

        target = cv2.imread(target_path)
        source = cv2.imread(source_path)

        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize
        source = cv2.resize(source, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, dsize=self.size, interpolation=cv2.INTER_CUBIC)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = self.default_prompt
        if self.class_hint:
            prompt = prompt + " of " + ', '.join(self.classes[file_name])

        return dict(jpg=target, txt=prompt, hint=source, name=file_name)
    

class PascalScribbleDataset(Dataset):
    def __init__(
            self, 
            size: Tuple[int, int] = (512, 512), # Input image size
            train: bool = True, # Whether to use training split or not
            overfit: bool = False, # Whether to limit the dataset to 10 images
            class_hint: bool = False, # Whether to include class names in the CLIP prompt
            split_path: Optional[str] = None, # Path to the SSL split
            one_hot_labels: bool = False, # Whether to use one-hot labels instead of RGB for hints
        ):
        self.DATASET_PATH = '/path/to/datasets/PascalVOC'
        
        with open(os.path.join(self.DATASET_PATH, 'train.txt' if train else 'val.txt'), 'r') as f:
            self.image_names = f.readlines()
        # Remove new line character
        self.image_names = list(map(lambda n: n[:-1], self.image_names))

        if split_path is not None:
            new_image_names = []
            with open(os.path.join(self.DATASET_PATH, split_path), 'r') as f:
                for line in f.readlines():
                    id = line.split(' ')[0].split('/')[-1].split('.')[0]
                    if id in self.image_names:
                        new_image_names.append(id)
            self.image_names = new_image_names

        # Get present classes
        self.class_hint = class_hint
        if class_hint:
            with open(f"{self.DATASET_PATH}/classes.json", 'r') as f:
                self.classes = json.load(f)
        else:
            self.classes = None
        
        self.size = size
        self.one_hot_labels = one_hot_labels
        self.default_prompt = "a high-quality, detailed, and professional image"
        if overfit:
            self.image_names = self.image_names[:10]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict:
        file_name = self.image_names[idx]
        target_path = f"{self.DATASET_PATH}/JPEGImages/{file_name}.jpeg"

        target = cv2.imread(target_path)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.one_hot_labels:
            source_path = f"{self.DATASET_PATH}/pascal_2012_scribble/{file_name}.png"
            source = plt.imread(source_path)
            # Use nearest neigbour upscaling so that labels are correct
            source = cv2.resize(source, dsize=self.size, interpolation=cv2.INTER_NEAREST)
            source = torch.tensor(source * 255, dtype=torch.int64)
            source[source > 20] = 21
            source = torch.nn.functional.one_hot(source, num_classes=22).float()
        else:
            source_path = f"{self.DATASET_PATH}/pascal_2012_scribble_color_coded/{file_name}.png"
            source = cv2.imread(source_path)
            # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = cv2.resize(source, dsize=self.size, interpolation=cv2.INTER_CUBIC)
            source = source.astype(np.float32) / 255.0

        prompt = self.default_prompt
        if self.class_hint:
            prompt = prompt + " of " + ', '.join(self.classes[file_name])

        return dict(jpg=target, txt=prompt, hint=source, name=file_name)


def get_dataloaders(config: ExpConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds: Dataset
    val_ds: Dataset

    if config.dataset.value == DatasetEnum.PascalSegmentation.value:
        train_ds = PascalSegmentationDataset(size=tuple(config.image_size), train=True, overfit=config.overfit)
        val_ds = PascalSegmentationDataset(size=tuple(config.image_size), train=False, overfit=config.overfit)
        assert len(train_ds) + len(val_ds) == 12031 or config.overfit

    elif config.dataset.value == DatasetEnum.PascalScribble.value:
        train_ds = PascalScribbleDataset(size=tuple(config.image_size), train=True, 
                                         overfit=config.overfit, class_hint=config.class_hint, 
                                         one_hot_labels=config.one_hot_labels,
                                         split_path=config.split_path)
        val_ds = PascalScribbleDataset(size=tuple(config.image_size), train=False, 
                                       overfit=config.overfit, class_hint=config.class_hint,
                                       one_hot_labels=config.one_hot_labels)
        assert len(train_ds) + len(val_ds) == 12031 or config.overfit or config.split_path

    else:
        raise NotImplementedError(f"{config.dataset} not added")
    
    print(f"Train dataset has {len(train_ds)} images")
    print(f"Val dataset has {len(val_ds)} images")
    train_dataloader = DataLoader(train_ds, num_workers=config.num_workers, 
                                  batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, num_workers=config.num_workers, 
                                batch_size=config.batch_size, shuffle=False)
    return train_dataloader, val_dataloader
