import sys
import torch
from data import PascalSegmentationDataset, ADE20KDataset
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/home/jacob/scribblegen/ControlNet')
from ControlNet.cldm.model import create_model, load_state_dict

device = torch.device('cuda:2')
dataset = PascalSegmentationDataset(train=True)

model_config_path = './ControlNet/models/cldm_v15.yaml'
model_checkpoint_path = 'logs/PascalSegmentation/lightning_logs/version_15/checkpoints/epoch=126-step=84073.ckpt'

data_path = 'temp_data/'

model = create_model(model_config_path).to(device)
model.load_state_dict(load_state_dict(model_checkpoint_path, location=device))
model.cond_stage_model.device = device

for i, data in enumerate(dataset):
    data['hint'] = torch.tensor(data['hint'], device=device).unsqueeze(0)
    data['jpg'] = torch.tensor(data['jpg'], device=device).unsqueeze(0)
    data['txt'] = [data['txt']]
    x, c = model.get_input(data, k=0)
    log = model.log_images(data)

    img = log['samples_cfg_scale_9.00']
    img = torch.clamp(img, -1., 1.)
    img = img.cpu().numpy()[0]
    img = np.moveaxis(img, 0, -1)
    img = (img + 1) / 2
    img = (img * 255).astype(np.uint8)
    img = img[...,::-1]

    path = data_path + data['name'] + '.jpeg'
    print(f'Saving new image to {path} {i}/{len(dataset)}')
    plt.imsave(path, img, format='jpeg')
    