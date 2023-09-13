import tensorflow as tf

from .crf_computation import CRFComputation
from .crf_config import CRFConfig


class CRF(tf.Module):
    def __init__(self, config: CRFConfig, name=None):
        super().__init__(name)

        self.computation = CRFComputation(config)

        self.bilateral_blur_neighbors = None
        self.spatial_blur_neighbors = None

        print("Activate computational graph...")
        (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        ) = self.computation.init_partially(
            tf.ones(
                shape=[config.height, config.width, config.d_bifeats - 2],
                dtype=tf.float32,
            )
        )
        bilateral_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                bilateral_coords_1d_uniq,
                tf.range(self.computation.bilateral_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        bilateral_blur_neighbors = bilateral_hash_table.lookup(bilateral_ns) + 1

        spatial_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                spatial_coords_1d_uniq,
                tf.range(self.computation.spatial_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        spatial_blur_neighbors = spatial_hash_table.lookup(spatial_ns) + 1
        
        self.computation.mean_field_approximation(
            tf.ones(shape=[config.height, config.width, 1], dtype=tf.float32),
            bilateral_blur_neighbors=bilateral_blur_neighbors,
            spatial_blur_neighbors=spatial_blur_neighbors,
        )
        print("Graph activated.")

    def init(self, image: tf.Tensor) -> None:
        (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        ) = self.computation.init_partially(image)

        bilateral_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                bilateral_coords_1d_uniq,
                tf.range(self.computation.bilateral_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        self.bilateral_blur_neighbors = bilateral_hash_table.lookup(bilateral_ns) + 1

        spatial_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                spatial_coords_1d_uniq,
                tf.range(self.computation.spatial_vars.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        self.spatial_blur_neighbors = spatial_hash_table.lookup(spatial_ns) + 1

    def mean_field_approximation(self, unary: tf.Tensor) -> tf.Tensor:
        MAP = self.computation.mean_field_approximation(
            unary,
            bilateral_blur_neighbors=self.bilateral_blur_neighbors,
            spatial_blur_neighbors=self.spatial_blur_neighbors,
        )

        return MAP
