import time

import numpy as np
import matplotlib.pyplot as plt

from util.example_util import image2unary
from model_src.crf_config import CRFConfig
from model_src.crf import CRF

image_name = "../../data/examples/im2.png"
anno_name = "../../data/examples/anno2.png"
computation_path = "computation/crf"

unary, image, n_labels = image2unary(image_name, anno_name)
h, w, c = image.shape

crf = CRF(model_config=CRFConfig(), computation_path=computation_path)

start = time.time()
MAP = crf.inference(unary.astype(np.float32), image.astype(np.float32)).numpy()
print("Time: ", time.time() - start)
plt.imshow(MAP)
plt.show()