import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util.example_util import image2unary
from model_src.crf_config import CRFConfig
from model_src.crf_v2 import CRF

image_name = "../../data/examples/im1.png"
anno_name = "../../data/examples/anno1.png"
crf_computation_path = "./saved_model/crf_computation"

unary, image, n_labels = image2unary(image_name, anno_name)
h, w, c = image.shape

config = CRFConfig(tf.constant(h, dtype=tf.int32), tf.constant(w, dtype=tf.int32))

crf = CRF(config)

print("Activate CRF...")
crf.init(tf.constant(tf.ones(shape=image.shape, dtype=tf.float32)))
crf.mean_field_approximation(tf.constant(tf.ones(shape=unary.shape, dtype=tf.float32)))
print("CRF activated.")

start = time.time()
crf.init(tf.constant(image.astype(np.float32), dtype=tf.float32))
MAP = crf.mean_field_approximation(tf.constant(unary.astype(np.float32), dtype=tf.float32))
print("Time: ", time.time() - start)
plt.imshow(MAP.numpy())
plt.show()