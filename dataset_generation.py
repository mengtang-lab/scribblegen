import argparse
import sys
import torch
from data import PascalSegmentationDataset, PascalScribbleDataset, ADE20KDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

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
    parser.add_argument('--guidance-scale', type=float, default=9.,
                        help='scale between unconditioned and conditioned model output')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='number of inference steps')
    parser.add_argument('--encode-ratio', type=float, default=1.0,
                        help='level of noise to add to input image in range [0, 1]')
    parser.add_argument('--split', type=str, default=None,
                        help='path to split of images to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size to use for inference')
    args = parser.parse_args()

    print(args)

    device = torch.device(f'cuda:{args.gpu_id}')
    dataset = PascalScribbleDataset(train=True, class_hint=args.add_hint, split_path=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    out_dir = args.out_dir

    model_config_path = './ControlNet/models/cldm_v15.yaml'
    os.makedirs(out_dir, exist_ok=True)
    assert len(os.listdir(out_dir)) == 0

    model = create_model(model_config_path).to(device)
    embedding = None
    state_dict = load_state_dict(args.checkpoint, location=device)
    if 'drop_out_embedding' in state_dict.keys():
        embedding = state_dict['drop_out_embedding']
        del state_dict['drop_out_embedding']
    model.load_state_dict(state_dict)
    model.drop_out_embedding = embedding
    model.cond_stage_model.device = device
    model.eval()

    for data in tqdm(dataloader):
        log = model.log_images(
            data, N=len(data['jpg']),
            unconditional_guidance_scale=args.guidance_scale,
            ddim_steps=args.num_steps,
            noise_level=args.encode_ratio,
        )

        imgs = log[f'samples_cfg_scale_{args.guidance_scale:.2f}']
        imgs = torch.clamp(imgs, -1., 1.)
        imgs = imgs.cpu().numpy()
        imgs = np.moveaxis(imgs, 1, -1)
        imgs = (imgs + 1) / 2
        imgs = (imgs * 255).astype(np.uint8)
        imgs = imgs[...,::-1]
        print(imgs.shape)

        for i, img in enumerate(imgs):
            path = os.path.join(out_dir, data['name'][i] + '.jpeg')
            plt.imsave(path, img, format='jpeg')

if __name__ == "__main__":
   main()
