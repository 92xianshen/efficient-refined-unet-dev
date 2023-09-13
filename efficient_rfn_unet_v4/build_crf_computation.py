import tensorflow as tf

from model_src.crf_computation import CRFComputation

export_dir = "computation/crf"

computation = CRFComputation()

tf.saved_model.save(computation, export_dir=export_dir)
print("Write to {}.".format(export_dir))