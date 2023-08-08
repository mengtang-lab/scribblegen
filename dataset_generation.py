import argparse
import sys
import torch
from data import PascalSegmentationDataset, PascalScribbleDataset, ADE20KDataset
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.append('/home/jacob/scribblegen/ControlNet')
from ControlNet.cldm.model import create_model, load_state_dict

def main():
    # Set these args:
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--checkpoint', type=str, default=None, required=True,
                        help='path of the model checkpoint for inference')
    parser.add_argument('--out-dir', type=str, default=None, required=True,
                        help='directory for output images')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='which gpu to use for inference')
    parser.add_argument('--dataset', type=str, default='PascalScribble',
                        choices=['ADE20K', 'PascalSegmentation', 'PascalScribble'],
                        help='dataset to run inference on')
    parser.add_argument('--add-hint', action='store_true', default=False,
                        help='whether to add class hints to prompts')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')
    dataset = PascalScribbleDataset(train=True, class_hint={args.add_hint})
    out_dir = args.out_dir

    model_config_path = './ControlNet/models/cldm_v15.yaml'
    os.makedirs(out_dir, exist_ok=True)
    assert len(os.listdir(out_dir)) == 0

    model = create_model(model_config_path).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, location=device))
    model.cond_stage_model.device = device

    for i, data in enumerate(dataset):
        data['hint'] = torch.tensor(data['hint'], device=device).unsqueeze(0)
        data['jpg'] = torch.tensor(data['jpg'], device=device).unsqueeze(0)
        data['txt'] = [data['txt']]
        log = model.log_images(data)

        img = log['samples_cfg_scale_9.00']
        img = torch.clamp(img, -1., 1.)
        img = img.cpu().numpy()[0]
        img = np.moveaxis(img, 0, -1)
        img = (img + 1) / 2
        img = (img * 255).astype(np.uint8)
        img = img[...,::-1]

        path = os.path.join(out_dir, data['name'] + '.jpeg')
        print(f'Saving new image to {path} {i}/{len(dataset)}')
        plt.imsave(path, img, format='jpeg')

if __name__ == "__main__":
   main()
