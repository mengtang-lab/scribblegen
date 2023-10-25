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

        # Get paths
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

        # Set up augmentation arguments
        if self.split[0] == 'train':
            self.aug_scheme = args.aug_scheme
            self.aug_datasets = args.aug_datasets.split(',')
            self.aug_datasets = [ds.strip() for ds in self.aug_datasets]
            self.replacement_prob = args.replacement_prob
            self.curriculum_epochs = None
            self.curriculum_ds = 0
            if args.curriculum is not None:
                self.curriculum_epochs = [int(epoch.strip()) for epoch in args.curriculum.split(',')]
                assert len(self.curriculum_epochs) == len(self.aug_datasets)
                assert self.aug_scheme != "sample", "Curriculum learning doesn't work with random sampling"
            
            if args.aug_best_dict is not None:
                path = os.path.join(self._base_dir, args.aug_best_dict)
                assert os.path.isfile(path), f"No json file at {path}"
                with open(path, 'r') as f:
                    selection_dict = json.load(f)
            else:
                selection_dict = None
        else:
            self.aug_scheme = None
            self.aug_datasets = []
            self.curriculum_epochs = None
        self._epoch = 0

        _splits_dir = os.path.join(self._base_dir, 'SSL_splits')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.full_categories = []
        self.dataset_nums = [] # -1 for real and i for ith synthetic dataset

        self.real_data = []
        self.synth_data = []

        for splt in self.split:
            # Get the image IDs
            if args.ssl_split is not None and splt == "train":
                split_path = os.path.join(_splits_dir, args.ssl_split, 'labeled.txt')
            else:
                split_path = os.path.join(_splits_dir, splt + '.txt')
            lines = []
            with open(split_path, "r") as f:
                for line in f.readlines():
                    id = line.split(' ')[0].split('/')[-1].split('.')[0]
                    lines.append(id)

            for line in lines:
                if args.aug_scheme != 'synth-only' or splt == 'val':
                    # Add real data
                    _image = os.path.join(self._image_dir, line + ".jpg")
                    _cat = os.path.join(self._cat_dir, line + ".png")
                    _full_cat = os.path.join(self._full_cat_dir, line + ".png")
                    assert os.path.isfile(_image), f'no image at {_image}'
                    assert os.path.isfile(_cat), f'no label at {_cat}'
                    assert os.path.isfile(_full_cat), f'no label at {_full_cat}'
                    self.real_data.append((line, _image, _cat, _full_cat, -1))

                if split == 'train' and args.aug_scheme != 'none':
                    # Add synthetic data
                    self.synth_data.append(self._add_synthetic(line, selection_dict, args.aug_best_k))

        # Display stats
        print(f'Number of images in {split}: {len(self)} ({len(self.real_data)} are real)')


    def _add_synthetic(self, line, selection_dict=None, best_k=1):
        # Returns list of (image id, image path, label, full label, dataset used)
        if self.aug_scheme == 'none':
            return []
        data = []
        for i, ds in enumerate(self.aug_datasets):
            label_type = 'scribble' if self.args.scribbles else 'full'
            if selection_dict is not None:
                if i > best_k:
                    break
                ds = selection_dict[line][i]
            synth_dir = os.path.join(self._synth_dir, f'{label_type}_{ds}')
            _id = line
            _image = os.path.join(synth_dir, line + ".jpeg")
            _cat = os.path.join(self._cat_dir, line + ".png")
            _full_cat = os.path.join(self._full_cat_dir, line + ".png")
            assert os.path.isfile(_image), f'no image at {_image}'
            assert os.path.isfile(_cat), f'no label at {_cat}'
            assert os.path.isfile(_full_cat), f'no label at {_full_cat}'
            data.append((_id, _image, _cat, _full_cat, ds))
        return data


    def __len__(self):
        if self.aug_scheme == 'none' or self.aug_scheme == 'replacement' or self.split[0] != 'train':
            return len(self.real_data)
        elif self.curriculum_epochs is not None and self.aug_scheme == 'append':
            return len(self.real_data) + len(self.synth_data)
        elif self.aug_scheme == 'append':
            return len(self.real_data) + len(self.synth_data) * len(self.aug_datasets)
        elif self.aug_scheme == 'sample':
            return len(self.real_data) + len(self.synth_data)
        elif self.aug_scheme == 'synth-only':
            return len(self.synth_data) * len(self.aug_datasets)
        else:
            raise NotImplementedError()

    @property
    def epoch(self):
        return self._epoch
    
    @epoch.setter
    def epoch(self, val):
        self._epoch = val
        idx = 0
        while idx != len(self.curriculum_epochs) and self.epoch >= self.curriculum_epochs[idx]:
            idx += 1
        if idx != self.curriculum_ds:
            self.curriculum_ds = idx - 1
            print(f"Now using dataset {self.curriculum_ds}: {self.aug_datasets[self.curriculum_ds]}")


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
        image_index = None
        dataset_index = None

        if self.aug_scheme == 'none' or self.split[0] == 'val':
            use_real = True

        elif self.aug_scheme == 'append':
            if index < len(self.real_data):
                use_real = True
            else:
                use_real = False
                image_index = index % len(self.real_data)
                if self.curriculum_epochs is not None:
                    dataset_index = self.curriculum_ds
                else:
                    dataset_index = (index // len(self.real_data)) - 1

        elif self.aug_scheme == 'sample':
            if index < len(self.real_data):
                use_real = True
            else:
                use_real = False
                image_index = index % len(self.real_data)
                dataset_index = torch.randint(len(self.aug_datasets), (1, )).item()

        elif self.aug_scheme == 'synth-only':
            use_real = False
            image_index = index % len(self.synth_data)
            dataset_index = index // len(self.synth_data)

        elif self.aug_scheme == 'replacement':
            prob = torch.rand((1, )).item()
            if prob <= self.replacement_prob:  # Replace with synth image
                use_real = False
                image_index = index
                if self.curriculum_epochs is not None:
                    dataset_index = self.curriculum_ds
                else:  # Otherwise sample which dataset to use
                    dataset_index = torch.randint(len(self.aug_datasets), (1, )).item()
            else:  # Don't replace
                use_real = True

        else:
            raise NotImplementedError()

        if use_real:
            _id, _img_path, _target_path, _full_target_path, _dataset = self.real_data[index]
            dataset_index = -1
        else:
            _id, _img_path, _target_path, _full_target_path, _dataset = self.synth_data[image_index][dataset_index]
        _img = Image.open(_img_path).convert('RGB')
        _target = Image.open(_target_path)
        _full_target = Image.open(_full_target_path)

        return _img, _target, _full_target, _id, dataset_index, _img_path

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


