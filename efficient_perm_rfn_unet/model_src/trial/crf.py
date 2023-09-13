"""
`CRFComputation` includes all static computations. 
Should be built once before use.
"""

import tensorflow as tf
from .crf_config import CRFConfig
from .permutohedralx_computation_dynamic import PermutohedralXComputation


class CRF(tf.Module):
    def __init__(self, config: CRFConfig, name=None):
        super().__init__(name)

        self.config = config

        self.bilateral_computation = PermutohedralXComputation(
            N=self.config.n_feats, d=self.config.d_bifeats
        )
        self.spatial_computation = PermutohedralXComputation(
            N=self.config.n_feats, d=self.config.d_spfeats
        )
        self.bilateral_blur_neighbors = None
        self.spatial_blur_neighbors = None

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    #     ]
    # )
    def partial_init(self, image: tf.Tensor) -> tf.Tensor:
        # - Create bilateral features.
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        ys, xs = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij")  # [h, w]
        ys_bifeats, xs_bifeats = (
            tf.cast(ys, dtype=tf.float32) / self.config.theta_alpha,
            tf.cast(xs, dtype=tf.float32) / self.config.theta_alpha,
        )
        color_feats = image / self.config.theta_beta
        bilateral_feat = tf.concat(
            [xs_bifeats[..., tf.newaxis], ys_bifeats[..., tf.newaxis], color_feats],
            axis=-1,
        )
        bilateral_feat = tf.reshape(bilateral_feat, shape=[-1, self.config.d_bifeats])

        # - Create spatial features.
        ys_spfeats, xs_spfeats = (
            tf.cast(ys, dtype=tf.float32) / self.config.theta_gamma,
            tf.cast(xs, dtype=tf.float32) / self.config.theta_gamma,
        )
        spatial_feat = tf.concat(
            [xs_spfeats[..., tf.newaxis], ys_spfeats[..., tf.newaxis]], axis=-1
        )
        spatial_feat = tf.reshape(spatial_feat, shape=[-1, self.config.d_spfeats])

        bilateral_coords_1d_uniq, bilateral_ns = self.bilateral_computation.init(
            bilateral_feat
        )
        spatial_coords_1d_uniq, spatial_ns = self.spatial_computation.init(spatial_feat)

        return (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        )

    def init(self, image: tf.Tensor) -> tf.Tensor:
        (
            bilateral_coords_1d_uniq,
            bilateral_ns,
            spatial_coords_1d_uniq,
            spatial_ns,
        ) = self.partial_init(image)

        bilateral_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                bilateral_coords_1d_uniq,
                tf.range(self.bilateral_computation.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        self.bilateral_blur_neighbors = bilateral_hash_table.lookup(bilateral_ns) + 1

        spatial_hash_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                spatial_coords_1d_uniq,
                tf.range(self.spatial_computation.M, dtype=tf.int32),
            ),
            default_value=-1,
        )
        self.spatial_blur_neighbors = spatial_hash_table.lookup(spatial_ns) + 1

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    #     ]
    # )
    def mean_field_approximation(self, unary: tf.Tensor) -> tf.Tensor:
        unary_shape = tf.shape(unary)  # [H, W, C] kept
        height, width, num_classes = unary_shape[0], unary_shape[1], unary_shape[2]
        # - Compute symmetric weights
        all_one = tf.ones(shape=[self.config.n_feats, 1], dtype=tf.float32)  # [N, 1]
        bilateral_norm_val = self.bilateral_computation.compute(
            all_one, self.bilateral_blur_neighbors
        )
        bilateral_norm_val = 1.0 / (bilateral_norm_val**0.5 + 1e-20)
        spatial_norm_val = self.spatial_computation.compute(
            all_one, self.spatial_blur_neighbors
        )
        spatial_norm_val = 1.0 / (spatial_norm_val**0.5 + 1e-20)

        # - Initialize Q
        unary = tf.reshape(
            unary, shape=[self.config.n_feats, num_classes]
        )  # flatten, [N, C]
        Q = tf.nn.softmax(-unary, axis=-1)  # [N, C]

        for i in range(self.config.num_iterations):
            tmp1 = -unary  # [N, C]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = self.bilateral_computation.compute(
                Q * bilateral_norm_val, self.bilateral_blur_neighbors
            )  # [N, C]
            bilateral_out *= bilateral_norm_val  # [N, C]

            # - Symmetric normalization and spatial message passing
            spatial_out = self.spatial_computation.compute(
                Q * spatial_norm_val, self.spatial_blur_neighbors
            )  # [N, C]
            spatial_out *= spatial_norm_val  # [N, C]

            # - Message passing
            message_passing = (
                self.config.bilateral_compat * bilateral_out
                + self.config.spatial_compat * spatial_out
            )  # [N, C]

            # - Compatibility transform
            pairwise = self.config.compatibility * message_passing  # [N, C]

            # - Local update
            tmp1 -= pairwise  # [N, C]

            # - Normalize
            Q = tf.nn.softmax(tmp1)  # [N, C]

        # - Maximum posterior
        Q = tf.reshape(Q, shape=unary_shape)  # [H, W, C]
        MAP = tf.math.argmax(Q, axis=-1)  # [H, W]

        return MAP
