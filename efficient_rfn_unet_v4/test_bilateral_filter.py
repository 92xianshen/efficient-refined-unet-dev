# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
from model_src.bilateral_filter import BilateralHighDimFilter
from model_src.bilateral_filter_config import BilateralHighDimFilterConfig
import matplotlib.pyplot as plt
import cv2
import time

computation_path = "computation/bilateral_filter"

# - Load data
im = Image.open("../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

h, w, n_chs = im.shape

# - Default config
default_config = BilateralHighDimFilterConfig(range_sigma=.25, space_sigma=5)
bilateral_filter = BilateralHighDimFilter(
    model_config=default_config, computation_path=computation_path
)

# Initialize filter
print("Start...")
start = time.time()
bilateral_filter.init(im)
print("Filter initialized.")

all_ones = np.ones((h, w, 1), dtype=np.float32)
norms = bilateral_filter.compute(all_ones)
norms = norms.numpy()
# norms = norms.numpy().reshape((h, w, 1))

src = im
dst = bilateral_filter.compute(src.astype(np.float32))
dst = dst.numpy()
dst = dst / norms
dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
print("Time:", time.time() - start)

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.waitKey()
