import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from numpy import float32
from torch.utils.data import DataLoader

from dataset import ClimateHackDataset
from loss import MS_SSIMLoss
from submission.model import Model

plt.rcParams["figure.figsize"] = (20, 12)