import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Define the MRIDataset class
class MRIDataset(Dataset):
    def __init__(self, df, img_transform=None, mask_transform=None):
        self.df = df.reset_index(drop=True)
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        mask_path = self.df.loc[idx, "mask_path"]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask).convert("L")

        if self.img_transform:   img = self.img_transform(img)
        if self.mask_transform:  mask = self.mask_transform(mask)

        return img, mask


def get_loaders(
    data_dir: str,
    batch_size: int,
    img_size: int,
    num_workers: int,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str = "diagnosis"):
    """
    Prepare PyTorch DataLoaders for MRI images and masks.

    Args:
        data_dir (str): Path to the root data directory. Subdirectories represent classes.
        batch_size (int): Number of samples per batch.
        img_size (int): Size to resize images and masks to (square).
        num_workers (int): Number of subprocesses for data loading.
        test_size (float): Proportion of data to reserve for the test split.
        random_state (int): Seed for reproducible train/test split.
        stratify_col (str): Column name to stratify split on (default: 'diagnosis').

    Returns:
        train_loader, test_loader: PyTorch DataLoader instances.
    """
    # --- LOAD DATA ---
    records = []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            records.append((cls, fpath))

    df = pd.DataFrame(records, columns=["dir_name", "image_path"])
    df_imgs = df[~df["image_path"].str.contains("mask")]
    df_masks = df[df["image_path"].str.contains("mask")]

    imgs = sorted(
        df_imgs["image_path"].values,
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    masks = sorted(
        df_masks["image_path"].values,
        key=lambda x: int(os.path.basename(x).split("_")[-2])
    )

    df_final = pd.DataFrame({
        "image_path": imgs,
        "mask_path": masks,
    })

    def diagnosis(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return int(np.max(mask) != 0)

    df_final["diagnosis"] = df_final["mask_path"].apply(diagnosis)

    # --- Transforms ---
    img_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # --- Split data ---
    train_df, test_df = train_test_split(
        df_final,
        test_size=test_size,
        random_state=random_state,
        stratify=df_final[stratify_col]
    )

    # --- Create datasets ---
    train_ds = MRIDataset(train_df, img_transform=img_transforms, mask_transform=mask_transforms)
    test_ds = MRIDataset(test_df, img_transform=img_transforms, mask_transform=mask_transforms)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

