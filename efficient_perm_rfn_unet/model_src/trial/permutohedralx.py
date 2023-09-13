import tensorflow as tf
from .permutohedralx_computation import PermutohedralXComputation


class PermutohedralX(tf.Module):
    def __init__(self, N: tf.int32, d: tf.int32, name: str = None):
        super().__init__(name)

        self.computation = PermutohedralXComputation(
            tf.constant(N, dtype=tf.int32), tf.constant(d, dtype=tf.int32)
        )
        self.blur_neighbors = None

    def init(self, feature: tf.Tensor) -> None:
        coords_1d_uniq, ns = self.computation.init(feature)
        hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                coords_1d_uniq, tf.range(self.computation.M, dtype=tf.int32)
            ),
            default_value=-1,
        )
        self.blur_neighbors = hash_table.lookup(ns) + 1

    def compute(self, inp: tf.Tensor, reverse: tf.bool = False) -> tf.Tensor:
        return self.computation.compute(inp, self.blur_neighbors, reverse=reverse)
