import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import albumentations
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ACDCDataset(Dataset):
    """Automated Cardiac Diagnosis Challenge (ACDC) Dataset"""
    def __init__(self, root_dir, dataset='training', sequence=False, transform_ind=True):
        self.root_dir = root_dir
        self.img_path_list = []
        self.img_name_list = []
        self.dataset = dataset
        self.sequence = sequence
        self.transform_ind = transform_ind
        self.transform = albumentations.Compose([
            albumentations.augmentations.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)]
        )

        self.img_dir = os.path.join(self.root_dir, f'{self.dataset}/labeled_-1/images/')
        self.msk_all_dir = os.path.join(self.root_dir, f'{self.dataset}/labeled_-1/masks_all/')
        self.msk_dir = os.path.join(self.root_dir, f'{self.dataset}/labeled_-1/masks/')

        for img in os.listdir(self.img_dir):
            self.img_path_list.append(os.path.join(self.img_dir, img))
            self.img_name_list.append(os.path.splitext(img)[0])

        print(f'Loaded {len(self.img_path_list)} {self.dataset} images')

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_name_list[idx]
        msk_all_name = img_name
        msk_lv_name = img_name + '_LV'
        msk_rv_name = img_name + '_RV'
        msk_myo_name = img_name + '_MYO'

        img_path = self.img_path_list[idx]
        msk_all_path = os.path.join(self.msk_all_dir, f'{msk_all_name}.png')
        msk_lv_path = os.path.join(self.msk_dir, f'{msk_lv_name}.png')
        msk_rv_path = os.path.join(self.msk_dir, f'{msk_rv_name}.png')
        msk_myo_path = os.path.join(self.msk_dir, f'{msk_myo_name}.png')

        # Read images and masks corresponding to a given index
        image = io.imread(img_path, as_gray=True)
        msk_all = io.imread(msk_all_path, as_gray=True)
        msk_lv = io.imread(msk_lv_path, as_gray=True)
        msk_rv = io.imread(msk_rv_path, as_gray=True)
        msk_myo = io.imread(msk_myo_path, as_gray=True)

        # For sequence modeling, add a temporal dimension
        if self.sequence:
            image = torch.from_numpy(image).float().unsqueeze(0)

        masks = torch.stack([
            torch.zeros((224, 224), dtype=torch.float32),  # Create mask for the background
            torch.tensor(msk_lv, dtype=torch.float32),
            torch.tensor(msk_rv, dtype=torch.float32),
            torch.tensor(msk_myo, dtype=torch.float32)
        ],
            dim=0
        )

        # Apply transformations
        if self.transform_ind:
            augmented_image = self.transform(image=image)
            image = torch.from_numpy(augmented_image['image'])

        sample = {'image': np.expand_dims(image, axis=-1),
                  # 'masks_all': msk_all,
                  'masks_all': np.expand_dims(msk_all, axis=-1),
                  'masks': masks, 'idx': idx}

        return sample
