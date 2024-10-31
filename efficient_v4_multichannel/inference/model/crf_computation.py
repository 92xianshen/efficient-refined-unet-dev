import tensorflow as tf

from .bilateral_filter_computation_multichannel import BilateralHighDimFilterComputation
from .spatial_filter_computation_v2 import SpatialHighDimFilterComputation


class CRFComputation(tf.Module):
    def __init__(self, config, name=None):
        super().__init__(name)

        self.config = config
        # - Initialize
        self.bilateral_filter = BilateralHighDimFilterComputation(
            height=self.config.height, 
            width=self.config.width, 
            n_channels=self.config.n_channels, 
            range_sigma=self.config.theta_beta, 
            space_sigma=self.config.theta_alpha, 
            range_padding=self.config.bilateral_range_padding,
            space_padding=self.config.bilateral_space_padding,
            n_iters=self.config.n_iters,
        )
        self.spatial_filter = SpatialHighDimFilterComputation(
            height=self.config.height, 
            width=self.config.width, 
            space_sigma=self.config.theta_gamma, 
            space_padding=self.config.spatial_space_padding, 
            n_iters=self.config.n_iters, 
        )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of `unary`, [H, W, N], float32
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # of `image`, [H, W, N], float32
        # tf.TensorSpec(shape=[], dtype=tf.float32), # of `theta_alpha`, [], float32
        # tf.TensorSpec(shape=[], dtype=tf.float32), # of `theta_beta`, [], float32
        # tf.TensorSpec(shape=[], dtype=tf.float32), # of `theta_gamma`, [], float32
        # tf.TensorSpec(shape=[], dtype=tf.float32), # of `bilateral_compat`, [], float32
        # tf.TensorSpec(shape=[], dtype=tf.float32), # of `spatial_compat`, [], float32
        # tf.TensorSpec(shape=[], dtype=tf.float32), # of `compatibility`, [], float32
        # tf.TensorSpec(shape=[], dtype=tf.int32), # of `num_iterations`, [], int32
    ])
    def mean_field_approximation(
        self,
        unary: tf.Tensor,
        image: tf.Tensor,
        # theta_alpha: tf.float32,
        # theta_beta: tf.float32,
        # theta_gamma: tf.float32,
        # bilateral_compat: tf.float32,
        # spatial_compat: tf.float32,
        # compatibility: tf.float32,
        # num_iterations: tf.int32,
    ) -> tf.Tensor:
        # unary_shape = tf.shape(unary)
        # height, width, num_classes = unary_shape[0], unary_shape[1], unary_shape[2]
        # n_channels = tf.shape(image)[2]
        (
            bilateral_splat_coords,
            bilateral_data_size,
            bilateral_data_shape,
            bilateral_slice_idx,
            bilateral_alpha_prod,
        ) = self.bilateral_filter.init(image)

        (
            spatial_splat_coords,
            spatial_data_size,
            spatial_data_shape,
            spatial_slice_idx,
            spatial_alpha_prod,
        ) = self.spatial_filter.init()

        # - Compute symmetric weights
        all_ones = tf.ones([self.config.height, self.config.width, 1], dtype=tf.float32)  # [H, W, 1]
        bilateral_norm_vals = self.bilateral_filter.compute(
            all_ones,
            splat_coords=bilateral_splat_coords,
            data_size=bilateral_data_size,
            data_shape=bilateral_data_shape,
            slice_idx=bilateral_slice_idx,
            alpha_prod=bilateral_alpha_prod,
        )
        bilateral_norm_vals = 1.0 / (bilateral_norm_vals**0.5 + 1e-20)
        spatial_norm_vals = self.spatial_filter.compute(
            all_ones,
            splat_coords=spatial_splat_coords,
            data_size=spatial_data_size,
            data_shape=spatial_data_shape,
            slice_idx=spatial_slice_idx,
            alpha_prod=spatial_alpha_prod,
        )
        spatial_norm_vals = 1.0 / (spatial_norm_vals**0.5 + 1e-20)

        # - Initialize Q
        Q = tf.nn.softmax(-unary, axis=-1)  # [H, W, N]

        for i in range(self.config.num_iterations):
            tmp1 = -unary  # [H, W, N]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = self.bilateral_filter.compute(
                Q * bilateral_norm_vals,
                splat_coords=bilateral_splat_coords,
                data_size=bilateral_data_size,
                data_shape=bilateral_data_shape,
                slice_idx=bilateral_slice_idx,
                alpha_prod=bilateral_alpha_prod,
            )
            bilateral_out *= bilateral_norm_vals

            # - Symmetric normalization and spatial message passing
            spatial_out = self.spatial_filter.compute(
                Q * spatial_norm_vals,
                splat_coords=spatial_splat_coords,
                data_size=spatial_data_size,
                data_shape=spatial_data_shape,
                slice_idx=spatial_slice_idx,
                alpha_prod=spatial_alpha_prod,
            )
            spatial_out *= spatial_norm_vals

            # - Message passing
            message_passing = (
                self.config.bilateral_compat * bilateral_out + self.config.spatial_compat * spatial_out
            )  # [H, W, C]

            # - Compatibility transform
            pairwise = self.config.compatibility * message_passing  # [N, C]

            # - Local update
            tmp1 -= pairwise  # [N, C]

            # - Normalize
            Q = tf.nn.softmax(tmp1)  # [N, C]

        # - Maximum posterior estimation
        MAP = tf.math.argmax(Q, axis=-1)  # [H, W]

        return MAP
