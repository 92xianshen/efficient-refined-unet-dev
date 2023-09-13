# -*- coding: utf-8 -*-
"""
Compare the efficiency performance of bilateral filter v1 and v3.
V1 has a pure computation graph, whereas v3 has inner variables.

No significant difference observed, therefore proceed with v3.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

# - Import v1
from model_src.bilateral_filter import BilateralHighDimFilter
from model_src.bilateral_filter_config import BilateralHighDimFilterConfig

computation_path_v1 = "computation/bilateral_filter"
computation_path_v3 = "computation/bilateral_filter_v3"

# - Load data
im = Image.open("../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

h, w, n_chs = im.shape

# - Create v1 and v3
default_config_v1 = BilateralHighDimFilterConfig(range_sigma=.25, space_sigma=16)
bilateral_filter_v1 = BilateralHighDimFilter(model_config=default_config_v1, computation_path=computation_path_v1)
bilateral_filter_v3 = tf.saved_model.load(computation_path_v3)

# - Run v1
print("Start v1...")
start = time.time()
for _ in range(100):
    bilateral_filter_v1.init(im)

    all_ones = np.ones((h, w, 1), dtype=np.float32)
    norms = bilateral_filter_v1.compute(all_ones)
    norms = norms.numpy()

    src = im
    dst = bilateral_filter_v1.compute(src.astype(np.float32))
    dst = dst.numpy()
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
print("Time of v1 100 filterings:", time.time() - start)

# Run v3
print("Start v3...")
start = time.time()
for _ in range(100):
    bilateral_filter_v3.init(im, range_sigma=tf.constant(0.25, dtype=tf.float32), space_sigma=tf.constant(16, dtype=tf.float32))

    all_ones = np.ones((h, w, 1), dtype=np.float32)
    norms = bilateral_filter_v3.compute(all_ones)
    norms = norms.numpy()

    src = im
    dst = bilateral_filter_v3.compute(src.astype(np.float32))
    dst = dst.numpy()
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
print("Time of v3 100 filterings:", time.time() - start)

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.waitKey()
