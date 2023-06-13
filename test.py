import sys
import logging
import time

import cv2
import numpy as np
import pickle as pkl
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from random import randint
from torch.utils.data import Dataset
from ADE20K.utils.utils_ade20k import loadAde20K

sys.path.append('/home/jacob/work/ControlNet')

class ADE20KDataset(Dataset):
    def __init__(self, size=(512, 512)):
        self.DATASET_PATH = 'ADE20K/dataset'
        index_file = 'index_ade20k.pkl'
        with open('{}/{}'.format(self.DATASET_PATH, index_file), 'rb') as f:
            self.index_ade20k = pkl.load(f)
        
        self.size = size
        self.default_prompt = "a high-quality, detailed, and professional image"

    def __len__(self):
        return len(self.index_ade20k['filename'])

    def __getitem__(self, idx):
        full_file_name = '{}/{}'.format(self.index_ade20k['folder'][idx], self.index_ade20k['filename'][idx])
        try:
            info = loadAde20K('{}/{}'.format(self.DATASET_PATH, full_file_name))
        except UnicodeDecodeError as e:
            logging.error(f"Error loading image {idx} at {full_file_name}")
            raise e
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

dataset = ADE20KDataset()

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './ControlNet/models/control_sd15_ini.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    precision=32, 
    callbacks=[logger],
    accumulate_grad_batches=2,
    max_epochs=1,
    check_val_every_n_epoch=1, 
    default_root_dir='/home/jacob/work/ControlNet',
)


# Train!
trainer.fit(model, dataloader)
