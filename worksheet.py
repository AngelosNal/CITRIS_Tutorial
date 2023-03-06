## Standard libraries
import os
import sys
import numpy as np
import random
from PIL import Image
from types import SimpleNamespace

## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

# Import CITRIS-VAE
sys.path.append('CITRIS')
from pytorch_lightning import LightningModule
# from CITRIS.models.citris_vae import CITRISVAE
from CITRIS.models.citris_nf import CITRISNF

CHECKPOINT_PATH = "./"

pretrained_CITRIS_path = os.path.join(CHECKPOINT_PATH, "citris" + ".ckpt")

if os.path.isfile(pretrained_CITRIS_path):
    print(f"Found pretrained model at {pretrained_CITRIS_path}, loading...")
    model = CITRISNF.load_from_checkpoint(pretrained_CITRIS_path)
