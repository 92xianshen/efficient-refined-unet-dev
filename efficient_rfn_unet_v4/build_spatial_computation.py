import tensorflow as tf

from model_src.spatial_filter_computation import SpatialHighDimFilterComputation

export_dir = "computation/spatial_filter"

computation = SpatialHighDimFilterComputation()

tf.saved_model.save(computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))