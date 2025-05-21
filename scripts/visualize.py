

# --- Import Libraries ---
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import seaborn as sns
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from scripts.data import get_loaders


# Define device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# --- Set Plot Style ---
sns.set_style("dark")
plt.style.use("dark_background")


# --- Define Parameters ---
BATCH_SIZE = 64
IMG_SIZE = 32
NUM_WORKERS = 0

train_loader, test_loader = get_loaders("data/", batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS)

# --- Visualize some images and masks ---
def visualize_data(loader, num_images=5):
    images, masks = next(iter(loader))
    images = images.numpy()
    masks = masks.numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(10, 20))
    for i in range(num_images):
        axes[0, i].imshow(images[i].transpose(1, 2, 0))
        axes[0, i].axis("off")
        axes[1, i].imshow(masks[i].transpose(1, 2, 0))
        axes[1, i].axis("off")
    plt.subplots_adjust(hspace=-0.9)
    plt.savefig("plots/images.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # Visualize the data
    visualize_data(train_loader, num_images=5)
    print("Images and masks saved to plots/images.png")


