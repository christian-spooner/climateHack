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
import timeit
import sys
import cv2
import numpy as np
from imageio import imread
from PIL import Image


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def flowIm(im1, im2):
    im1 = np.float64(im1)
    im2 = np.float64(im2)
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    hsv = draw_hsv(flow)
    im2w = warp_flow(im2, flow)
    return im2w


'''
fig, ax = plt.subplots(1, 3, figsize=(15,3))
for i, d in enumerate([im1, im2, im2w]):
    ax[i].imshow(d, cmap='viridis')
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()

cv2.imwrite("tmp/flow.jpg",hsv)
cv2.imwrite("tmp/im1.jpg", im1)
cv2.imwrite("tmp/im2.jpg", im2)
cv2.imwrite("tmp/im2w.jpg", im2w)
'''
