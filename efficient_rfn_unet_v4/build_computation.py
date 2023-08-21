import tensorflow as tf

from model_src.bilateral_filter_computation import BilateralHighDimFilterComputation

export_dir = "computation/bilateral_filter"

computation = BilateralHighDimFilterComputation()

tf.saved_model.save(computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))