import os
import torch
import numpy as np
import albumentations as A
from skimage import io
from torch.utils.data import Dataset


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ACDCDataset(Dataset):
    """Automated Cardiac Diagnosis Challenge (ACDC) Dataset"""
    def __init__(self, root_dir, dataset='training', transform_ind=True, transform_mask=False,
                 crop_resize_ind=False, margin=0.02, target_size=(224, 224)):
        self.root_dir = root_dir
        self.img_path_list = []
        self.img_name_list = []
        self.dataset = dataset
        self.transform_ind = transform_ind
        self.transform_mask = transform_mask
        self.crop_resize_ind = crop_resize_ind
        self.margin = margin
        self.target_size = target_size
        self.transform_common = A.Compose([
            # Geometric augmentations
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Elastic deformation to simulate variations in anatomy
            A.ElasticTransform(p=0.2, alpha=1.0, sigma=50, alpha_affine=50),

            # Blurring to simulate low-resolution artifacts
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.GaussianBlur(blur_limit=3, p=0.1),
                ],
                p=0.3,
            ),
        ],
            additional_targets={"mask": "mask"}  # To apply the same transformations to masks
        )

        self.transform_img = A.Compose([
            # Brightness and contrast adjustments
            A.RandomBrightnessContrast(p=0.2),

            # Adding Gaussian noise
            A.GaussNoise(p=0.2),

            # Normalize image
            A.augmentations.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),
        ],
        )


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
        # msk_all_name = img_name
        msk_lv_name = img_name + '_LV'
        msk_rv_name = img_name + '_RV'
        msk_myo_name = img_name + '_MYO'

        img_path = self.img_path_list[idx]
        msk_lv_path = os.path.join(self.msk_dir, f'{msk_lv_name}.png')
        msk_rv_path = os.path.join(self.msk_dir, f'{msk_rv_name}.png')
        msk_myo_path = os.path.join(self.msk_dir, f'{msk_myo_name}.png')

        # Read images and masks corresponding to a given index
        image = io.imread(img_path, as_gray=True)
        image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
        org_image = image.copy()
        msk_lv = io.imread(msk_lv_path, as_gray=True)
        msk_rv = io.imread(msk_rv_path, as_gray=True)
        msk_myo = io.imread(msk_myo_path, as_gray=True)

        msk_lv = (msk_lv > 0).astype(np.uint8)
        msk_rv = (msk_rv > 0).astype(np.uint8)
        msk_myo = (msk_myo > 0).astype(np.uint8)

        # Initialize the combined mask with background
        msk_all = np.zeros_like(msk_lv, dtype=np.uint8)

        # Assign 1 for LV, 2 for RV, and 3 for MYO
        msk_all[msk_lv == 1] = 1
        msk_all[msk_rv == 1] = 2
        msk_all[msk_myo == 1] = 3
        org_msk_all = msk_all.copy()

        # Apply transformations to image and mask
        if self.transform_ind:
            # Apply common transformations
            transformed = self.transform_common(image=image, mask=msk_all)
            image = transformed['image']
            msk_all = torch.from_numpy(transformed['mask'])

            # Apply image specific transformations
            transformed = self.transform_img(image=image)
            image = torch.from_numpy(transformed['image'])

            if self.transform_mask:
                # Apply median filtering to combined mask
                transform_mask = A.Compose([
                    A.MedianBlur(blur_limit=(7,9), p=1.0),
                ])
                transformed_msk_all = transform_mask(image=msk_all.to(torch.int8))
                msk_all = torch.from_numpy(transformed_msk_all['image'])

        # Create sample - additional channel dimension is added last for images
        sample = {
            'org_image': torch.from_numpy(org_image).unsqueeze(-1),
            'image': image.unsqueeze(-1),
            'org_masks_all': org_msk_all,
            'masks_all': msk_all,
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
