# Parts of this code were adapted from:
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# - https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

from os import path
from torch.utils.data import Dataset
import random
from pathlib import Path
import cv2
import albumentations as A
import albumentations.pytorch

import torch


def get_transform(args, mean, std) -> A.Compose:

    transformations = []
    # TODO check if We can also just remove all the A.Sequential ...
    if args.distort:
        transformations.append(A.Sequential([A.GridDistortion(p=0.3)]))
    if args.rotate:
        transformations.append(A.Sequential([A.geometric.rotate.Rotate()]))
    if args.img_dim is not None:
        transformations.append(A.Sequential([A.RandomCrop(width=args.img_dim, height=args.img_dim)]))
    if args.h_flip:
        transformations.append(A.Sequential([A.HorizontalFlip()]))
    if args.v_flip:
        transformations.append(A.Sequential([A.VerticalFlip()]))
    if args.brightness is not None or args.contrast is not None:
        transformations.append(A.Sequential([A.RandomBrightnessContrast(p=0.2, brightness_limit=args.brightness if args.brightness is not None else 0, contrast_limit=args.contrast if args.contrast is not None else 0)]))

    transformations.append(A.Normalize(mean=mean, std=std))
    transformations.append(A.ToFloat())
    transformations.append(A.pytorch.ToTensorV2())

    return A.Compose(transformations)


class ETHCILDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, split):
        assert split in ['train_only', 'train_split', 'val_split', 'stats', 'eval'], f'Split {split} does not exist in dataset.'
        assert path.exists(data_dir), f'Path {data_dir} does not exist.'
        
        self.SPLIT_PERCENTAGE = 0.2
        self.SEED = 42
        self.split = split
        self.mean = (0.5098, 0.5205, 0.5180)
        self.std = (0.2109, 0.2011, 0.1962)
        self.data_dir = Path(data_dir)


        if self.split in ['train_only', 'train_split', 'val_split', 'stats']:
            self.data_prefix = 'training'
        elif self.split == 'eval':
            self.data_prefix = 'test'

        self.images = sorted(list((Path(data_dir) / self.data_prefix / 'images').glob('*.png')))

        random.Random(42).shuffle(self.images)

        if self.split == 'train_split':
            self.images = self.images[int(len(self.images)*self.SPLIT_PERCENTAGE):]

        if self.split == 'val_split':
            self.images = self.images[:int(len(self.images)*self.SPLIT_PERCENTAGE)]

        if self.split in ['train_only', 'train_split']:
            self.transform = get_transform(mean=self.mean, std=self.std, args=args)
        elif self.split in ['val_split', 'eval']:
            self.transform = A.Compose([
                    A.CenterCrop(width=400, height=400),
                    A.Normalize(mean=self.mean, std=self.std),
                    A.ToFloat(),
                    A.pytorch.ToTensorV2()
                ])
        elif self.split in ['stats']:
            self.transform = A.Compose([
                    A.CenterCrop(width=400, height=400),
                    A.ToFloat(),
                    A.pytorch.ToTensorV2()
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        current_image_path = self.images[index]
        sat_img = cv2.imread(str(current_image_path))
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        if self.split in ['train_only', 'train_split', 'val_split']:

            current_label_path = self.data_dir / self.data_prefix / 'groundtruth' / current_image_path.name

            segmentation = cv2.imread(str(current_label_path), 0)

            sample = self.transform(image=sat_img, mask=segmentation)

            sat_img = sample['image']
            segmentation = sample['mask']

            # we map from [0,255] to [0, 1]
            segmentation = segmentation / 255

            return sat_img, segmentation

        elif self.split in ['stats', 'eval']:

            sat_img = self.transform(image=sat_img)['image']

            # the zero is a dummy label such that we are consistent in our API
            return sat_img, torch.tensor(0)


class ETHMultiCityDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, split, cities):
        assert split in ['train_only', 'train_split', 'val_split', 'stats', 'eval'], f'Split {split} does not exist in dataset.'
        assert path.exists(data_dir), f'Path {data_dir} does not exist.'
        for city in cities:
            assert city in ['paris', 'zurich', 'berlin', 'chicago'], f'City {city} does not exist.'

        self.mean = (0.3082, 0.3285, 0.3042)
        self.std = (0.1752, 0.1803, 0.1745)

        #self.mean = (0.5098, 0.5205, 0.5180)
        #self.std = (0.2109, 0.2011, 0.1962)

        self.split = split
        self.cities = cities

        # TODO: DOES THIS MAKE SENSE.. this was done by manually counting pixels
        self.scale = 34

        self.SPLIT_PERCENTAGE = 0.2
        self.images = []

        for city in self.cities:
            current_path = Path(data_dir) / city
            temp = sorted(list(current_path.glob('*_image.png')))
            random.Random(42).shuffle(temp)
            if self.split == 'train_split':
                temp = temp[int(len(temp) * self.SPLIT_PERCENTAGE):]

            if self.split == 'val_split':
                temp = temp[:int(len(temp) * self.SPLIT_PERCENTAGE)]

            self.images.extend(temp)

        if self.split in ['train_only', 'train_split']:
            self.transform = get_transform(mean=self.mean, std=self.std, args=args)
        elif self.split in ['val_split', 'eval']:
            self.transform = A.Compose([
                    A.CenterCrop(width=400, height=400),
                    A.Normalize(mean=self.mean, std=self.std),
                    A.ToFloat(),
                    A.pytorch.ToTensorV2()
                ])
        elif self.split in ['stats']:
            self.transform = A.Compose([
                    A.CenterCrop(width=400, height=400),
                    A.ToFloat(),
                    A.pytorch.ToTensorV2()
                ])

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        current_image_path = self.images[index]
        current_label_path = str(current_image_path).strip('_image.png') + '_labels.png'

        sat_img = cv2.imread(str(current_image_path))
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        segmentation = cv2.imread(str(current_label_path))
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)

        # the following lines zero out the blue component of everything that is not a street
        segmentation[:, :, 2][segmentation[:, :, 0] == 255] = 0
        segmentation[:, :, 2][segmentation[:, :, 1] == 255] = 0

        width = int(sat_img.shape[1] * self.scale / 100)
        height = int(sat_img.shape[0] * self.scale / 100)
        dim = (width, height)

        sat_img = cv2.resize(sat_img, dim, interpolation=cv2.INTER_AREA)
        segmentation = cv2.resize(segmentation, dim, interpolation=cv2.INTER_AREA)

        segmentation = segmentation[:, :, 2]

        sample = self.transform(image=sat_img, mask=segmentation)
        sat_img = sample['image']
        segmentation = sample['mask']

        # we map from [0,255] to [0, 1]
        segmentation = segmentation / 255

        return sat_img, segmentation


def create_datasets(args, root_path, split, datasets):
    dataset_list = []
    if 'cil' in datasets:
        dataset_list.append(ETHCILDataset(args=args, data_dir=str(root_path/'cil_data'), split=split))

    cities_to_add = list({'berlin', 'paris', 'zurich', 'chicago'}.intersection(set(datasets)))

    if len(cities_to_add) > 0:
        dataset_list.append(ETHMultiCityDataset(args=args, data_dir=str(root_path/'cities_data'), split=split, cities=cities_to_add))

    return dataset_list


def create_dataloaders(args):
    # first figure out which datasets we are dealing with
    train_set = set(args.train_data)
    val_set = set(args.val_data)

    train_only = list(train_set.difference(val_set))
    train_and_val = list(train_set.intersection(val_set))
    val_only = list(val_set.difference(train_set))

    current_path = Path(args.data_dir)

    train_datasets = create_datasets(args, current_path, 'train_only', train_only)
    train_datasets.extend(create_datasets(args, current_path, 'train_split', train_and_val))

    # TODO is this smart to use val_split here?
    val_datasets = create_datasets(args, current_path, 'val_split', val_only)
    val_datasets.extend(create_datasets(args, current_path, 'val_split', train_and_val))

    train_data = torch.utils.data.ConcatDataset(train_datasets)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    if len(val_datasets) > 0:
        eval_data = torch.utils.data.ConcatDataset(val_datasets)
        eval_data_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        return train_data_loader, eval_data_loader

    return train_data_loader, None
