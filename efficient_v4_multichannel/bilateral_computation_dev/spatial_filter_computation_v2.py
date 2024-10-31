"""
Bilateral high-dim filter, built with TF 2.x.

v2: __init__()
"""

import tensorflow as tf


class SpatialHighDimFilterComputation(tf.Module):
    def __init__(self, height, width, space_sigma, space_padding=2, n_iters=2, name=None):
        super().__init__(name)

        self.height, self.width = tf.constant(height, dtype=tf.int32), tf.constant(width, dtype=tf.int32)
        self.size = self.height * self.width
        self.dim = tf.constant(2, dtype=tf.int32)

        self.space_sigma = tf.constant(space_sigma, dtype=tf.float32)
        self.space_padding = tf.constant(space_padding, dtype=tf.int32)
        
        self.n_iters = tf.constant(n_iters, dtype=tf.int32)

    @tf.function(input_signature=None)
    def init(self) -> tf.Tensor:
        """
        Initialize, `feature` should be three-channel and channel-last
        """

        def clamp(
            x: tf.Tensor, min_value: tf.float32, max_value: tf.float32
        ) -> tf.Tensor:
            return tf.maximum(min_value, tf.minimum(max_value, x))

        def get_both_indices(coord: tf.Tensor, size: tf.int32) -> tf.Tensor:
            """
            Method to get left and right indices of slice interpolation.
            """
            left_index = clamp(
                tf.cast(coord, dtype=tf.int32), min_value=0, max_value=size - 1
            )
            right_index = clamp(left_index + 1, min_value=0, max_value=size - 1)
            return left_index, right_index

        # size = height * width
        # dim = 2

        # Height and width of data grid, scala, dtype int
        small_height = tf.cast(tf.cast(self.height - 1, tf.float32) / self.space_sigma, dtype=tf.int32) + 1 + 2 * self.space_padding
        small_width = tf.cast(tf.cast(self.width - 1, tf.float32) / self.space_sigma, dtype=tf.int32) + 1 + 2 * self.space_padding

        # Space coordinates, shape [H, W], dtype int
        yy, xx = tf.meshgrid(tf.range(self.height), tf.range(self.width), indexing="ij")  # [H, W]
        yy, xx = tf.cast(yy, dtype=tf.float32), tf.cast(xx, dtype=tf.float32)
        # Spatial coordinates of splat, shape [H, W]
        splat_yy = tf.cast(yy / self.space_sigma + 0.5, dtype=tf.int32) + self.space_padding
        splat_xx = tf.cast(xx / self.space_sigma + 0.5, dtype=tf.int32) + self.space_padding
        # Spatial coordinates of slice, shape [H, W]
        slice_yy = tf.cast(yy, dtype=tf.float32) / self.space_sigma + tf.cast(self.space_padding, dtype=tf.float32)
        slice_xx = tf.cast(xx, dtype=tf.float32) / self.space_sigma + tf.cast(self.space_padding, dtype=tf.float32)

        # Spatial interpolation index of slice
        y_index, yy_index = get_both_indices(slice_yy, small_height)  # [H, W]
        x_index, xx_index = get_both_indices(slice_xx, small_width)  # [H, W]

        # Spatial interpolation factor of slice
        y_alpha = tf.reshape(slice_yy - tf.cast(y_index, dtype=tf.float32), shape=[-1, ], )  # [H x W, ]
        x_alpha = tf.reshape(slice_xx - tf.cast(x_index, dtype=tf.float32), shape=[-1, ], )  # [H x W, ]

        # - Bilateral interpolation index and factor
        interp_indices = [
            y_index,
            yy_index,
            x_index,
            xx_index,
        ]  # [10, H x W]
        alphas = [
            1.0 - y_alpha,
            y_alpha,
            1.0 - x_alpha,
            x_alpha,
        ]  # [10, H x W]

        # Method of coordinate transformation
        def coord_transform(idx):
            return tf.reshape(idx[:, 0, :] * small_width + idx[:, 1, :], shape=[-1, ], )  # [2^d x H x W, ]

        # Initialize interpolation
        offset = tf.range(self.dim) * 2  # [d, ]
        # Permutation
        permutations = tf.stack(
            tf.meshgrid(
                tf.range(2),
                tf.range(2),
                indexing="ij",
            ),
            axis=-1,
        )
        permutations = tf.reshape(permutations, shape=[-1, self.dim])  # [2^d, d]
        permutations += offset[tf.newaxis, ...]
        permutations = tf.reshape(permutations, shape=[-1, ], )  # flatten, [2^d x d, ]
        alpha_prods = tf.reshape(tf.gather(alphas, permutations), shape=[-1, self.dim, self.size])  # [2^d, d, H x W]
        idx = tf.reshape(tf.gather(interp_indices, permutations), shape=[-1, self.dim, self.size])  # [2^d, d, H x W]

        # Shape and size of bialteral data grid
        data_shape = tf.stack([small_height, small_width, ], )
        data_size = small_height * small_width

        # Bilateral splat coordinates, shape [H x W, ]
        splat_coords = splat_yy * small_width + splat_xx
        splat_coords = tf.reshape(splat_coords, shape=[-1, ], )  # [H x W, ]

        # Interpolation indices and alphas of bilateral slice
        slice_idx = coord_transform(idx)
        alpha_prod = tf.math.reduce_prod(alpha_prods, axis=1)  # [2^d, H x W]

        return splat_coords, data_size, data_shape, slice_idx, alpha_prod

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # of `inp`, [H, W, N],
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),  # of `splat_coords`, [H x W, ], int32
            tf.TensorSpec(shape=[], dtype=tf.int32),  # of `data_size`, [], int32
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),  # of `data_shape`, [d, ], int32
            tf.TensorSpec(shape=[None, ], dtype=tf.int32),  # of `slice_idx`, [2^d x H x W, ], int32
            tf.TensorSpec(shape=[None, None, ], dtype=tf.float32),  # of `alpha_prod`, [H x W, ], float32
        ]
    )
    def compute(
        self,
        inp: tf.Tensor,
        splat_coords: tf.Tensor,
        data_size: tf.int32,
        data_shape: tf.Tensor,
        slice_idx: tf.Tensor,
        alpha_prod: tf.Tensor,
    ) -> tf.Tensor:
        # height, width = tf.shape(inp)[0], tf.shape(inp)[1]
        # size = height * width
        # dim = 2

        # Channel-last to channel-first because tf.map_fn
        inpT = tf.transpose(inp, perm=[2, 0, 1])

        def ch_filter(inp_ch: tf.Tensor) -> tf.Tensor:
            # Filter each channel
            inp_flat = tf.reshape(inp_ch, shape=[-1, ])  # [H x W, ]
            # ==== Splat ====
            data_flat = tf.math.bincount(
                splat_coords,
                weights=inp_flat,
                minlength=data_size,
                maxlength=data_size,
                dtype=tf.float32,
            )
            data = tf.reshape(data_flat, shape=data_shape)

            # ==== Blur ====
            buffer = tf.zeros_like(data)
            perm = [1, 0]

            for _ in range(self.n_iters):
                buffer, data = data, buffer

                for _ in range(self.dim):
                    # newdata = (buffer[:-2] + buffer[2:]) / 2.0
                    newdata = (buffer[:-2] + buffer[2:]) / 2.0 + buffer[1:-1] # bugfix
                    data = tf.concat([data[:1], newdata, data[-1:]], axis=0)
                    data = tf.transpose(data, perm=perm)
                    buffer = tf.transpose(buffer, perm=perm)

            del buffer

            # ==== Slice ====
            data_slice = tf.gather(tf.reshape(data, shape=[-1, ]), slice_idx)  # (2^dim x h x w)
            data_slice = tf.reshape(data_slice, shape=[-1, self.size])  # (2^dim, h x w)
            interpolations = alpha_prod * data_slice
            interpolation = tf.reduce_sum(interpolations, axis=0)
            interpolation = tf.reshape(interpolation, shape=[self.height, self.width])

            return interpolation

        outT = tf.map_fn(ch_filter, inpT)
        out = tf.transpose(outT, perm=[1, 2, 0])  # Channel-first to channel-last

        return out
