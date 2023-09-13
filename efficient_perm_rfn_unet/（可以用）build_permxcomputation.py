import tensorflow as tf

from model_src.permutohedralx_computation import PermutohedralXComputation

N = 512 * 512
d = 3 + 2
export_dir = "computation/permx"

computation = PermutohedralXComputation(N=N, d=d)

tf.saved_model.save(computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))