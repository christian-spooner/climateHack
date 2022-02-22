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
    chunks="auto",  # Load the data as a Dask array
)
im0 = dataset["data"].sel(time="2020-07-04 12:00").isel(x=slice(128, 256), y=slice(128, 256)).to_numpy()
im1 = dataset["data"].sel(time="2020-07-04 12:05").isel(x=slice(128, 256), y=slice(128, 256)).to_numpy()


images = [im0, im1]
for i in range(1, 25):
    newIm = flowIm(images[i-1], images[i])
    images.append(newIm)


output = []
for i in range(2, 26):
    output.append(images[i][32:96, 32:96])


'''
plt.rcParams["figure.figsize"] = (20, 12)
fig, ax = plt.subplots(1, 24, figsize=(15,3))
for i in range(0, 24):
    ax[i].imshow(output[i], cmap='viridis')
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()


stop = timeit.default_timer()
execution_time = stop - start
print("Time: " + str(execution_time))
'''
