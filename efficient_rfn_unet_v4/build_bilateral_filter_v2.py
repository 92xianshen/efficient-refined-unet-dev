import tensorflow as tf

from model_src.bilateral_filter_v2 import BilateralHighDimFilter

export_dir = "computation/bilateral_filter_v2"

computation = BilateralHighDimFilter(height=512, width=512, n_channels=3, range_sigma=.25, space_sigma=5, range_padding=2, space_padding=2)

tf.saved_model.save(computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))