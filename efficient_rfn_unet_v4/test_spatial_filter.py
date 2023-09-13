# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
from model_src.spatial_filter import SpatialHighDimFilter
from model_src.spatial_filter_config import SpatialHighDimFilterConfig
import matplotlib.pyplot as plt
import cv2
import time

computation_path = "computation/spatial_filter"

# - Load data
im = Image.open("../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

h, w, n_chs = im.shape

# - Default config
default_config = SpatialHighDimFilterConfig(space_sigma=3)
spatial_filter = SpatialHighDimFilter(
    model_config=default_config, computation_path=computation_path
)

# Initialize filter
print("Start...")
start = time.time()
spatial_filter.init(h, w)
print("Filter initialized.")

all_ones = np.ones((h, w, 1), dtype=np.float32)
norms = spatial_filter.compute(all_ones)
norms = norms.numpy()
# norms = norms.numpy().reshape((h, w, 1))

src = im
dst = spatial_filter.compute(src.astype(np.float32))
dst = dst.numpy()
dst = dst / norms
dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
print("Time:", time.time() - start)

# - TF depthwise conv
r = int(3 * 4 + .5)
x, y = np.mgrid[-r:r + 1, -r:r + 1]
k = np.exp(-(x ** 2 + y ** 2) / (2 * 3 ** 2), dtype=np.float32)
k[r, r] = 0

start2 = time.time()
src = tf.transpose(src.astype(np.float32), perm=[2, 0, 1])
src = tf.pad(src, paddings=[[0, 0], [r, r], [r, r]], mode="constant", constant_values=0)
ext_k = tf.pad(k, paddings=[[0, h - 1], [0, w - 1]], mode="constant", constant_values=0)
src_fft, k_fft = tf.signal.rfft2d(src), tf.signal.rfft2d(ext_k)
dst_fft = src_fft * k_fft[tf.newaxis, ...]
dst2 = tf.signal.irfft2d(dst_fft)
dst2 = dst2[..., 2 * r:, 2 * r:]
dst2 = tf.transpose(dst2, perm=[1, 2, 0])
dst2 = dst2.numpy()
dst2 = (dst2 - dst2.min()) / (dst2.max() - dst2.min() + 1e-5)
print("Time2:", time.time() - start2)

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.imshow("dst2", dst2[..., ::-1])

cv2.waitKey()
