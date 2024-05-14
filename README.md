# ScribbleGen: Generative Data Augmentation Improves Scribble-supervised Semantic Segmentation (CVPR2024 SyntaGen Workshop)
[Jacob Schnell](https://student.cs.uwaterloo.ca/~jschnell/), [Jieke Wang](), [Lu Qi](http://luqi.info/), [Vincent Tao Hu](https://taohu.me/), [Meng Tang](http://mengtang.org)

[arXiv](https://arxiv.org/abs/2311.17121) [Openreview](https://openreview.net/forum?id=0lJq8pmlXM)

```
@article{scribblegen,
      title={ScribbleGen: Generative Data Augmentation Improves Scribble-supervised Semantic Segmentation}, 
      author={Jacob Schnell and Jieke Wang and Lu Qi and Vincent Tao Hu and Meng Tang},
      year={2024},
      eprint={2311.17121},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Structure
Our codebase is comprised of three other codebases. Namely, our diffusion model based image synthesizer is built using [ControlNet](https://github.com/lllyasviel/ControlNet), our main method for training weakly supervised segmentation models is using [RLoss](https://github.com/mengtang-cv/rloss/tree/master), and our secondary method for training weakly supervised segmentation models is using [TreeEnergyLoss](https://github.com/megvii-research/TreeEnergyLoss). We include all the files required from each library in our codebase, but some files may be modified or removed.

# Environments

To download the required Python libraries and initialize the conda environment, run the following:
```
conda create --name scribblegen
conda activate scribblegen
conda config --add channels pytorch
conda config --add channels conda-forge
conda install --file requirements.txt
```
You may need to install `opencv-python` and `opencv-contrib-python` via pip.

# Dataset

To download the PascalScribble dataset used in our paper, run
```
cd ./data/pascal_scribble/
sh fetchPascalScribble.sh
cd ../VOC2012
sh fetchVOC2012.sh
```
You will need to update the paths in `rloss/pytorch/pytorch-deeplab_v3_plus/mypath.py` and `data.py`.

# Train ControlNet

The ControlNet image synthesizer can be trained on Pascal scribbles using the following command:
```
python train.py --config control_PascalScribble
```

# Training Data Synthesis

Training data can be synthesized once the ControlNet model has been trained by
```
python dataset_generation.py --checkpoint /path/to/controlnet.ckpt --out-dir data/pascal_scribble/JPEGImages_synthetic/scribble_synthetic/ --gpu-id 0 
```

# Training RLoss

An RLoss model can be trained using the following command:
```
cd rloss/pytorch/pytorch-deeplab_v3_plus
python train_withdensecrfloss.py --backbone resnet --lr 0.003 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-resnet --eval-interval 2 --dataset pascal --save-interval 10 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme append --aug-datasets synthetic --gpu-ids 1
```
