# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
from model_src.permutohedralx import PermutohedralX
import matplotlib.pyplot as plt
import cv2
import time

# computation_path = "computation/permx"

# - Load data
im = Image.open("../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

# - Create bilateral features
h, w, n_channels = im.shape
N = h * w
d = 2 + n_channels

invSpatialStdev = 1.0 / 5.0
invColorStdev = 1.0 / 0.25

# - Initialize lattice
lattice = PermutohedralX(N=tf.constant(N), d=tf.constant(d))
toyfeat = np.zeros(shape=(N, d), dtype=np.float32)
lattice.init(toyfeat)
lattice.compute(toyfeat)
print("lattice activated.")
print("Start...")
start = time.time()
init_time = 0
compute_time = 0

for _ in range(1):
    init_start = time.time()
    color_feat = tf.constant(im * invColorStdev, dtype=tf.float32)
    ys, xs = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
    ys, xs = (
        tf.cast(ys, dtype=tf.float32) * invSpatialStdev,
        tf.cast(xs, dtype=tf.float32) * invSpatialStdev,
    )
    feature = tf.concat([xs[..., tf.newaxis], ys[..., tf.newaxis], color_feat], axis=-1)
    feature = tf.reshape(feature, shape=[-1, 5])
    lattice.init(feature)
    init_time += time.time() - init_start

    compute_start = time.time()
    all_ones = np.ones((N, 1), dtype=np.float32)
    norms = lattice.compute(tf.constant(all_ones))
    norms = norms.numpy().reshape((h, w, 1))

    src = im.reshape((-1, n_channels))
    dst = lattice.compute(tf.constant(src, dtype=tf.float32))
    dst = dst.numpy().reshape((h, w, n_channels))
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)
    compute_time += time.time() - compute_start


print("Init time:", init_time)
print("Compute time:", compute_time)
print("Time:", time.time() - start)

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.waitKey()
