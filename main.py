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
from flow import flowIm
import timeit
import sys
import numpy as np

start = timeit.default_timer()
DATA_PATH = "./eumetsat_seviri_hrv_uk.zarr"
dataset = xr.open_dataset(
    DATA_PATH, 
    engine="zarr",
    chunks="auto",
)

















stop = timeit.default_timer()
execution_time = stop - start
print("Time: " + str(execution_time))