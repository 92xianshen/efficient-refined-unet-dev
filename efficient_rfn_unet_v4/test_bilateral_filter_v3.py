# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
# from model_src.bilateral_filter import BilateralHighDimFilter
import matplotlib.pyplot as plt
import cv2
import time

computation_path = "computation/bilateral_filter_v3"

# - Load data
im = Image.open("../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

h, w, n_chs = im.shape

# - Default config
bilateral_filter = tf.saved_model.load(computation_path)

# Initialize filter
print("Start...")
start = time.time()
bilateral_filter.init(im, range_sigma=tf.constant(0.25, dtype=tf.float32), space_sigma=tf.constant(16, dtype=tf.float32))
# bilateral_filter.init(im, range_sigma=0.25, space_sigma=16)
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
