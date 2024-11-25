import os
import torch
import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ACDCDataset(Dataset):
    """Automated Cardiac Diagnosis Challenge (ACDC) Dataset"""
    def __init__(self, root_dir, dataset='training', sequence=False, transform_ind=True,
                 crop_resize_ind=False, margin=0.02, target_size=(224, 224)):
        self.root_dir = root_dir
        self.img_path_list = []
        self.img_name_list = []
        self.dataset = dataset
        self.sequence = sequence
        self.transform_ind = transform_ind
        self.crop_resize_ind = crop_resize_ind
        self.margin = margin
        self.target_size = target_size
        self.transform = albumentations.Compose([
            albumentations.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Applies Gaussian blur with a kernel size between 3 and 7
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
        org_image = image.copy()
        msk_all = io.imread(msk_all_path, as_gray=True)
        org_msk_all = msk_all.copy()
        msk_lv = io.imread(msk_lv_path, as_gray=True)
        msk_rv = io.imread(msk_rv_path, as_gray=True)
        msk_myo = io.imread(msk_myo_path, as_gray=True)

        # For sequence modeling, add a temporal dimension
        if self.sequence:
            image = torch.from_numpy(image).float().unsqueeze(0)
            org_image = torch.from_numpy(org_image).float().unsqueeze(0)

        masks = torch.stack([
            torch.zeros(msk_lv.shape, dtype=torch.float32),  # Create mask for the background
            torch.tensor(msk_lv, dtype=torch.float32),
            torch.tensor(msk_rv, dtype=torch.float32),
            torch.tensor(msk_myo, dtype=torch.float32)
        ],
            dim=0
        )

        # Crop and resize transform
        if self.crop_resize_ind:
            # Compute the bounding box
            bounding_box, _ = self.get_combined_bounding_box_with_margin(masks, margin=self.margin)
            x_min, y_min, x_max, y_max = bounding_box

            crop_resize_transform = albumentations.Compose([
                albumentations.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                albumentations.Resize(self.target_size[0], self.target_size[1])
                # ToTensorV2()
            ],
            additional_targets={'mask_combined': 'mask'}
            )

            crop_resize_transformed = crop_resize_transform(image=image,
                                                            masks=[masks[i].numpy() for i in range(masks.shape[0])],
                                                            mask_combined=msk_all)
            image = crop_resize_transformed["image"]
            masks = torch.stack([torch.tensor(m) for m in crop_resize_transformed["masks"]])
            msk_all = crop_resize_transformed["mask_combined"]

            bounding_box = torch.tensor([
                x_min / self.target_size[0], y_min / self.target_size[0],
                x_max / self.target_size[0], y_max / self.target_size[0]
            ],
                dtype=torch.float32
            )
        else:
            bounding_box = None

        # Apply other transformations
        if self.transform_ind:
            # Transform cropped/ uncropped image
            augmented_image = self.transform(image=image)
            image = torch.from_numpy(augmented_image['image'])

            # Transform original image
            augmented_org_image = self.transform(image=org_image)
            org_image = torch.from_numpy(augmented_org_image['image'])

        sample = {
            'org_image': np.expand_dims(org_image, axis=-1),
            'image': np.expand_dims(image, axis=-1),
            # 'masks_all': msk_all,
            'org_masks_all': np.expand_dims(org_msk_all, axis=-1),
            'masks_all': np.expand_dims(msk_all, axis=-1),
            'masks': masks,
            'bounding_box': bounding_box,
            'idx': idx
        }

        return sample


    @staticmethod
    def get_combined_bounding_box_with_margin(stacked_masks, margin=0.02):
        """
        Compute a single bounding box with a margin that encompasses all non-zero regions across stacked binary masks.
        Args:
            stacked_masks: PyTorch tensor of shape (N, H, W) with binary values (0 or 1),
                           where N is the number of masks.
            margin: Float, margin to add to the bounding box as a fraction of width/height.
        Returns:
            bounding_box: [x_min, y_min, x_max, y_max] in normalized coordinates.
        """
        combined_mask = torch.any(stacked_masks, dim=0).int()  # Combine masks into a single mask

        if torch.sum(combined_mask) == 0:
            # Default bounding box when no ROI is present
            return [0.0, 0.0, 1.0, 1.0], combined_mask
        else:
            rows = torch.any(combined_mask, dim=1)
            cols = torch.any(combined_mask, dim=0)

            y_min, y_max = torch.where(rows)[0][[0, -1]]
            x_min, x_max = torch.where(cols)[0][[0, -1]]

            # Add margin
            height, width = combined_mask.shape
            y_min = max(0, y_min.item() - int(margin * height))
            y_max = min(height, y_max.item() + int(margin * height))
            x_min = max(0, x_min.item() - int(margin * width))
            x_max = min(width, x_max.item() + int(margin * width))

            bounding_box = [x_min, y_min, x_max, y_max]
            return bounding_box, combined_mask
