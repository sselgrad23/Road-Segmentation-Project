# Parts of this code were adapted from:
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# - https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

from os import path
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from pathlib import Path
import cv2
import albumentations as A
import albumentations.pytorch

import torch


def get_transform(args, std, mean) -> A.Compose:

    transformations = []
    # TODO check if We can also just remove all the A.Sequential ...
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

    transformations.append(A.ToFloat())
    transformations.append(A.Normalize(mean=mean, std=std))
    transformations.append(A.pytorch.ToTensorV2())

    return A.Compose(transformations)


class ETHStreetSegmentationCities(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', args=None):
        super(ETHStreetSegmentationCities, self).__init__()
        assert split in ['train', 'test', 'val', 'statistics'], f'Split {split} does not exist in dataset.'
        assert path.exists(root_dir), f'Path {root_dir} does not exist.'

        self.mean = (0.3082, 0.3285, 0.3042)
        self.std = (0.3082, 0.3285, 0.3042)

        self.split = split
        self.root_dir_path = Path(root_dir)
        # TODO: DOES THIS MAKE SENSE.. this was done by counting
        self.scale = 34

        # If you want to support validation on this set change here
        train_cities = ['paris', 'zurich','berlin', 'chicago']
        val_cities = []

        self.train_images = []
        self.val_images = []

        for city in train_cities:
            c_path = self.root_dir_path / city
            self.train_images.extend(list(c_path.glob('*_image.png')))

        for city in val_cities:
            c_path = self.root_dir_path / city
            self.val_images.extend(list(c_path.glob('*_image.png')))

        # keep the list of val images short
        self.test_images = self.val_images[:20]

        self.root_dir_path = Path(root_dir)

        self.train_transform = A.Compose([
            A.geometric.rotate.Rotate(),
            A.RandomCrop(width=400, height=400),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
            A.ToFloat(),
            A.Normalize(mean=self.mean, std=self.std),
            A.pytorch.ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.CenterCrop(width=400, height=400),
            A.ToFloat(),
            A.Normalize(mean=self.mean if self.split != 'statistics' else (0, 0, 0),
                        std=self.std if self.split != 'statistics' else (1, 1, 1)),
            A.pytorch.ToTensorV2()
        ])

    def __len__(self):
        if self.split in ['train', 'statistics']:
            return len(self.train_images)
        elif self.split == 'val':
            return len(self.val_images)
        else:
            raise Exception(f'Split {self.split} not defined.')

    def __getitem__(self, index):

        if self.split in ['train', 'val', 'statistics']:

            current_image_path = self.train_images[index] if self.split in ['train','statistics'] else self.val_images[index]

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

            transform = self.train_transform if self.split == 'train' else self.test_transform

            sample = transform(image=sat_img, mask=segmentation)
            sat_img = sample['image']
            segmentation = sample['mask']

            # we map from [0,255] to [0, 1]
            segmentation = (segmentation / (255))

            return sat_img, segmentation

        else:
            raise Exception(f'Split {self.split} not defined.')


class ETHEvalData(torch.utils.data.Dataset):

    def __init__(self, root_dir, args=None, split=None):
        self.split = split
        self.mean = (0.5187, 0.5266, 0.5200)
        self.std = (0.5187, 0.5266, 0.5200)
        self.root_dir_path = Path(root_dir)
        self.images = list((self.root_dir_path / 'images').glob('*.png'))
        self.transform = A.Compose([
            A.CenterCrop(width=400, height=400),
            A.ToFloat(),
            A.Normalize(mean=self.mean if self.split != 'statistics' else (0, 0, 0),
                        std=self.std if self.split != 'statistics' else (1, 1, 1)),
            A.pytorch.ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        current_image_path = self.images[index]

        sat_img = cv2.imread(str(current_image_path))
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        sat_img = self.transform(image=sat_img)['image']

        return sat_img


class ETHStreetSegmentation(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', args=None):
        self.split = split
        self.mean = (0.5100, 0.5208, 0.5185)
        self.std = (0.5100, 0.5208, 0.5185)
        assert split in ['train', 'test', 'val', 'statistics'], f'Split {split} does not exist in dataset.'
        assert path.exists(root_dir), f'Path {root_dir} does not exist.'
        if split in ['train', 'statistics']:
            self.data_prefix = 'training'
        elif split == 'val':
            self.data_prefix = 'test'
        self.root_dir_path = Path(root_dir)
        image_path = self.root_dir_path / self.data_prefix / 'images'
        self.images = list((self.root_dir_path / self.data_prefix / 'images').glob('*.png'))

        # if we are in val or statistics mode this is not needed
        if split == 'train':
            self.transform = get_transform(args, std=self.std, mean=self.mean)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        current_image_path = self.images[index]
        if self.split in ['train']:
            current_label_path = self.root_dir_path / self.data_prefix / 'groundtruth' / (current_image_path.name)

            sat_img = cv2.imread(str(current_image_path))
            sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

            segmentation = cv2.imread(str(current_label_path), 0)




            if self.transform:
                sample = self.transform(image=sat_img, mask=segmentation)
                sat_img = sample['image']
                segmentation = sample['mask']

            # we map from [0,255] to [0, 1]
            segmentation = (segmentation / (255))

            return sat_img, segmentation

        elif self.split in ['val', 'statistics']:
            current_label_path = self.root_dir_path / self.data_prefix / 'groundtruth' / (current_image_path.name)

            sat_img = cv2.imread(str(current_image_path))
            sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

            # transform = A.Compose([
            #     A.CenterCrop(width=400, height=400),
            #     A.ToFloat(),
            #     A.Normalize(mean= self.mean if self.split != 'statistics' else (0,0,0), std=self.std if self.split != 'statistics' else (1,1,1)),
            #     A.pytorch.ToTensorV2()
            # ])

            transform = A.Compose([
                A.Sequential([
                    A.CenterCrop(width=400, height=400),
                    A.ToFloat(),
                    A.pytorch.ToTensorV2()
                ]) if self.split == 'statistics' else
                A.Sequential([
                    A.CenterCrop(width=400, height=400),
                    A.ToFloat(),
                    A.Normalize(mean=self.mean, std=self.std),
                    A.pytorch.ToTensorV2()
                ])
            ])

            sat_img = transform(image=sat_img)['image']

            return sat_img, torch.tensor(1)

        else:
            raise Exception(f'Split {self.split} not defined.')

