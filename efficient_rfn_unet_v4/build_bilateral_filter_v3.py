import tensorflow as tf

from model_src.bilateral_filter_v3 import BilateralHighDimFilter

export_dir = "computation/bilateral_filter_v3"

computation = BilateralHighDimFilter(height=512, width=512, n_channels=3)

tf.saved_model.save(computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))