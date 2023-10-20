from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import json

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 debug=False, # Forces use of eval transform and adds extra info
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        if split == 'train':
            if not args.scribbles: # full supervision
                self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
                self._synth_dir = os.path.join(self._base_dir, 'JPEGImages_synthetic')
            else: # weak supervision with scribbles
                self._cat_dir = os.path.join(self._base_dir, 'pascal_2012_scribble')
                self._synth_dir = os.path.join(self._base_dir, 'JPEGImages_synthetic')
        elif split == 'val':
            self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._full_cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        self.debug = debug
        self.epoch = 0
        if self.split[0] == 'train':
            self.aug_scheme = args.aug_scheme
            self.aug_dataset = args.aug_dataset
            self.aug_ratio = args.aug_ratio
            self.aug_use_all = args.aug_use_all
            self.replacement_prob = args.replacement_prob
            if args.curriculum is not None:
                curriculum = json.loads(args.curriculum)
                epochs = list(curriculum.keys())
                sets = list(curriculum.values())
                sort_idx = np.argsort(epochs)
                self.curriculum_epochs = [int(epochs[idx]) for idx in sort_idx]
                self.curriculum_sets = [sets[idx] for idx in sort_idx]
                assert self.aug_ratio == 1
            else:
                self.curriculum_epochs = None
                self.curriculum_sets = None
            
            assert self.aug_dataset == 'normal' or (self.aug_ratio == 1 and not self.aug_use_all)
            if self.aug_scheme == 'best':
                path = os.path.join(self._base_dir, args.aug_best_dict)
                assert os.path.isfile(path), f"No json file at {path}"
                with open(path, 'r') as f:
                    selection_dict = json.load(f)
            else:
                selection_dict = None
        else:
            self.aug_scheme = False
            self.aug_dataset = None
            self.aug_ratio = 0
            self.aug_use_all = False

        _splits_dir = os.path.join(self._base_dir, 'SSL_splits')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.full_categories = []
        self.dataset_nums = [] # -1 for real and i for ith synthetic dataset

        self.data = []
        self.synth_data = []
        self.count = 0 # Number of instances

        for splt in self.split:
            if args.ssl_split is not None and splt == "train":
                split_path = os.path.join(_splits_dir, args.ssl_split, 'labeled.txt')
            else:
                split_path = os.path.join(_splits_dir, splt + '.txt')
            lines = []
            with open(split_path, "r") as f:
                for line in f.readlines():
                    id = line.split(' ')[0].split('/')[-1].split('.')[0]
                    lines.append(id)

            for ii, line in enumerate(lines):
                self.count += 1
                if args.aug_scheme != 'synth-only' or splt == 'val':
                    # Add real data
                    _image = os.path.join(self._image_dir, line + ".jpg")
                    _cat = os.path.join(self._cat_dir, line + ".png")
                    _full_cat = os.path.join(self._full_cat_dir, line + ".png")
                    assert os.path.isfile(_image), f'no image at {_image}'
                    assert os.path.isfile(_cat), f'no label at {_cat}'
                    assert os.path.isfile(_full_cat), f'no label at {_full_cat}'
                    self.data.append((line, _image, _cat, _full_cat, -1))

                # To use synthetic images too:
                if split == 'train' and args.aug_scheme != 'none':
                    self.synth_data.append(self._add_synthetic(line, selection_dict))

        # Display stats
        print('Number of images in {}: {:d} ({:d} are real)'.format(split, len(self), len(self.data)))


    def _add_synthetic(self, line, selection_dict=None):
        # Returns list of (image id, image path, label, full label, dataset used)
        if self.aug_scheme == 'none':
            return [], []
        data = []
        if self.aug_use_all:
            num_datasets = 8
        elif self.curriculum_sets is not None:
            num_datasets = len(self.curriculum_sets)
        else:
            num_datasets = self.aug_ratio
        for i in range(num_datasets):
            label_type = 'scribble' if self.args.scribbles else 'full'
            if self.curriculum_sets is None:
                if selection_dict is not None: # Only useful if aug_use_all = False
                    # Pick the dataset to use based on ordering in the selection dict
                    dataset = selection_dict[line][i]
                else:
                    # Naively pick ith synthetic image from ith dataset
                    dataset = i + 1
                synth_dir = os.path.join(self._synth_dir, f'{label_type}_{self.aug_dataset}_{dataset}')
            else:
                synth_dir = os.path.join(self._synth_dir, f'{label_type}_{self.curriculum_sets[i]}_1')
                dataset = i + 1
            _id = line
            _image = os.path.join(synth_dir, line + ".jpeg")
            _cat = os.path.join(self._cat_dir, line + ".png")
            _full_cat = os.path.join(self._full_cat_dir, line + ".png")
            assert os.path.isfile(_image), f'no image at {_image}'
            assert os.path.isfile(_cat), f'no label at {_cat}'
            assert os.path.isfile(_full_cat), f'no label at {_full_cat}'
            data.append((_id, _image, _cat, _full_cat, dataset))
        return data


    def __len__(self):
        if self.aug_scheme == 'none' or self.aug_scheme == 'replacement':
            return self.count
        elif self.aug_scheme == 'synth-only':
            return self.aug_ratio * self.count
        else:
            return (1 + self.aug_ratio) * self.count


    def __getitem__(self, index):
        _img, _target, _full_target, _id, _dataset, _path = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                if self.debug: # Add debugging information
                    data1 = self.transform_val(sample)
                    data2 = self.transform_val({'image': _img, 'label': _full_target})
                    return {'image': data1['image'], 'label': data1['label'], 'full_label': data2['label'], 
                            'id': _id, 'dataset': _dataset, 'path': _path}
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        if index < len(self.data) and self.aug_scheme != 'synth-only':
            if self.aug_scheme == 'replacement':
                prob = torch.rand((1, )).item()
                if prob <= self.replacement_prob:
                    if self.curriculum_epochs is not None:
                        instance_idx = 0
                        while instance_idx != len(self.curriculum_epochs) and self.epoch >= self.curriculum_epochs[instance_idx]:
                            instance_idx += 1
                        instance_idx -= 1
                    else:
                        instance_idx = 0
                    _id, _img_path, _target_path, _full_target_path, _dataset = self.synth_data[index][instance_idx]
                else:
                    _id, _img_path, _target_path, _full_target_path, _dataset = self.data[index]
            else:
                _id, _img_path, _target_path, _full_target_path, _dataset = self.data[index]
        elif self.aug_use_all:
            # img_idx is the index of the image of Pascal to load
            # instance_idx is the index of which instance of the synthetic recreation of that image to use
            img_idx = index % self.count
            instance_idx = torch.randint(8, (1, )).item()
            _id, _img_path, _target_path, _full_target_path, _dataset = self.synth_data[img_idx][instance_idx]
        else:
            img_idx = index % self.count
            instance_idx = index // self.count
            if self.curriculum_epochs is not None:
                instance_idx = 0
                while instance_idx != len(self.curriculum_epochs) and self.epoch >= self.curriculum_epochs[instance_idx]:
                    instance_idx += 1
                instance_idx -= 1
            elif self.aug_scheme != 'synth-only':
                instance_idx -= 1 # shift down since first `self.count` images are real not synth
            _id, _img_path, _target_path, _full_target_path, _dataset = self.synth_data[img_idx][instance_idx]
        _img = Image.open(_img_path).convert('RGB')
        _target = Image.open(_target_path)
        _full_target = Image.open(_full_target_path)

        return _img, _target, _full_target, _id, _dataset, _img_path

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


