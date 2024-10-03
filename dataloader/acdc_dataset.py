import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ACDCDataset(Dataset):
    """Automated Cardiac Diagnosis Challenge (ACDC) Dataset"""
    def __init__(self, root_dir, dataset='training', transform=None):
        self.root_dir = root_dir
        self.img_path_list = []
        self.img_name_list = []
        self.dataset = dataset
        self.transform = transform

        self.img_dir = os.path.join(self.root_dir, f'{self.dataset}/labeled_-1/images/')
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
        msk_lv_name = img_name + '_LV'
        msk_rv_name = img_name + '_RV'
        msk_myo_name = img_name + '_MYO'

        img_path = self.img_path_list[idx]
        msk_lv_path = os.path.join(self.msk_dir, f'{msk_lv_name}.png')
        msk_rv_path = os.path.join(self.msk_dir, f'{msk_rv_name}.png')
        msk_myo_path = os.path.join(self.msk_dir, f'{msk_myo_name}.png')

        # Read images and masks corresponding to a given index
        image = io.imread(img_path, as_gray=True)
        msk_lv = io.imread(msk_lv_path, as_gray=True)
        msk_rv = io.imread(msk_rv_path, as_gray=True)
        msk_myo = io.imread(msk_myo_path, as_gray=True)

        masks = torch.stack(
            [torch.tensor(msk_lv, dtype=torch.float32),
             torch.tensor(msk_rv, dtype=torch.float32),
             torch.tensor(msk_myo, dtype=torch.float32)],
            dim=0
        )

        sample = {'image': np.expand_dims(image, axis=-1), 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        sample['idx'] = idx

        return sample
