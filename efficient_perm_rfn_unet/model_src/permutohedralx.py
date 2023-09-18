import tensorflow as tf
from .permutohedralx_function import PermutohedralXFunction


class PermutohedralX(tf.Module):
    def __init__(self, N: tf.int32, d: tf.int32, name: str = None):
        super().__init__(name)

        self.computation = PermutohedralXFunction(
            tf.constant(N, dtype=tf.int32), tf.constant(d, dtype=tf.int32)
        )
        self.os = None
        self.ws = None
        self.blur_neighbors = None

    def init(self, feature: tf.Tensor) -> None:
        os, ws, coords_1d_uniq, M, ns = self.computation.init(feature)
        hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                coords_1d_uniq, tf.range(M, dtype=tf.int32)
            ),
            default_value=-1,
        )
        self.os = os
        self.ws = ws
        self.blur_neighbors = hash_table.lookup(ns) + 1
        self.M = M

    def compute(self, inp: tf.Tensor, reverse: tf.bool = False) -> tf.Tensor:
        return self.computation.compute(inp, os=self.os, ws=self.ws, blur_neighbors=self.blur_neighbors, M=self.M, reverse=reverse)
