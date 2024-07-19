# Cardiac MRI Segmentation of Ventricular Structures and Myocardium
This repository contains the code developed for cardiac MRI segmentation of ventricular structures and myocardium.

## Data Preparation
As the first step, 2D images slices should be extracted from the original NIfTI files. Run the below two commands to
generate training and test set images.

#### Training Set Images:
```
python data_prep_acdc.py --data_dir ./data/ACDC/original/training --save_data_dir ./data/ACDC/img_slices_v1/training
```

#### Test Set Images:
```
python data_prep_acdc.py --data_dir ./data/ACDC/original/testing --save_data_dir ./data/ACDC/img_slices_v1/testing
```
