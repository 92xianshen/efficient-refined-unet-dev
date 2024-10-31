"""
Bilateral high-dim filter, built with TF 2.x.
"""

import tensorflow as tf


class BilateralHighDimFilterComputation(tf.Module):
    def __init__(self, height, width, n_channels, range_sigma, space_sigma, range_padding=2, space_padding=2, n_iters=2, name=None):
        super().__init__(name)

        
        self.height, self.width, self.n_channels = height, width, n_channels
        self.size = self.height * self.width
        self.dim = self.n_channels + 2

        self.range_sigma, self.space_sigma = range_sigma, space_sigma

        self.range_padding, self.space_padding = range_padding, space_padding

        self.n_iters = n_iters

        # helper variables
        offset = tf.range(self.dim) * 2  # [d, ]
        dim_ranges = tf.tile([tf.range(2)], [self.dim, 1]) # [dim, 2]
        permutations = tf.stack(tf.meshgrid(*tf.unstack(dim_ranges), indexing="ij"), axis=-1)
        permutations = tf.reshape(permutations, shape=[-1, self.dim])  # [2^d, d]
        permutations += offset[tf.newaxis, ...]
        self.permutations = tf.reshape(permutations, shape=[-1, ])  # flatten, [2^d * d, ]

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # of `features`,  [H, W, C], float32, multi-channel and 3-rank
    #     ]
    # )
    def init(
        self,
        image: tf.Tensor
    ) -> tf.Tensor:
        """
        Initialize, `feature` should be three-channel and channel-last
        """

        def clamp(x: tf.Tensor, min_value: tf.float32, max_value: tf.float32) -> tf.Tensor:
            return tf.maximum(min_value, tf.minimum(max_value, x))

        def get_both_indices(coord: tf.Tensor, size: tf.int32) -> tf.Tensor:
            """
            Method to get left and right indices of slice interpolation.
            """
            left_index = clamp(tf.cast(coord, dtype=tf.int32), min_value=0, max_value=size - 1)
            right_index = clamp(left_index + 1, min_value=0, max_value=size - 1)
            return left_index, right_index

        # ==== Color transformation ====
        # - !!! Use the transpose of image, channel-first
        imageT = tf.transpose(image, perm=[2, 0, 1]) # [C, H, W], float32, channel-first

        print("imageT.shape: ", imageT.shape)

        ch_mins, ch_maxs = tf.reduce_min(imageT, axis=[1, 2]), tf.reduce_max(imageT, axis=[1, 2]) # [C, ], float32; [C, ], float32
        ch_deltas = ch_maxs - ch_mins # [C, ], float32
        chhs = imageT - ch_mins[:, tf.newaxis, tf.newaxis] # - Range coordinates, [C, H, W], dtype float

        small_chdepths = tf.cast(ch_deltas / self.range_sigma, dtype=tf.int32) + 1 + 2 * self.range_padding # depths of data grid [C, ] 

        splat_chhs = tf.cast(chhs / self.range_sigma + 0.5, dtype=tf.int32) + self.range_padding # range coordinates of splatting, [C, H, W]
        slice_chhs = chhs / self.range_sigma + tf.cast(self.range_padding, dtype=tf.float32) # range coordinates of slicing, [C, H, W]

        # - Slice interpolation range coordinates pairs
        ch_indices = tf.maximum(0, tf.minimum(small_chdepths[:, tf.newaxis, tf.newaxis] - 1, tf.cast(slice_chhs, dtype=tf.int32))) # [C, H, W]
        chh_indices = tf.maximum(0, tf.minimum(small_chdepths[:, tf.newaxis, tf.newaxis] - 1, ch_indices + 1)) # [C, H, W]

        ch_alphas = tf.reshape(slice_chhs - tf.cast(ch_indices, dtype=tf.float32), shape=[self.n_channels, -1]) # [C, H * W]

        # ==== Spatial transformation ====
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
        y_alpha = tf.reshape(slice_yy - tf.cast(y_index, dtype=tf.float32), shape=[-1, ]) # [H x W, ]
        x_alpha = tf.reshape(slice_xx - tf.cast(x_index, dtype=tf.float32), shape=[-1, ])  # [H x W, ]

        # - stack and transpose, [y_index, x_index, ..., yy_index, xx_index, ...] to [y_index, yy_index, x_index, xx_index, ...]
        spatial_indices = tf.stack([y_index, x_index], axis=0) # [2, H, W]
        spatiall_indices = tf.stack([yy_index, xx_index], axis=0) # [2, H, W]
        interp_indices = tf.concat([spatial_indices, ch_indices, spatiall_indices, chh_indices], axis=0) # [2 * d, H, W]
        interp_indices = tf.reshape(interp_indices, shape=[2 * self.dim, self.size]) # flatten along each dim, [d, H * W]
        interp_indices = tf.reshape(interp_indices, shape=[2, self.dim, self.size]) # [2, d, H * W]
        interp_indices = tf.transpose(interp_indices, perm=[1, 0, 2]) # [d, 2, H * W] 
        interp_indices = tf.reshape(interp_indices, shape=[2 * self.dim, self.size]) # [2 * d, H * W]
        
        # - Same as above
        spatial_alphas = tf.stack([y_alpha, x_alpha], axis=0) # [2, H * W]
        left_alphas = 1.0 - tf.concat([spatial_alphas, ch_alphas], axis=0) # [d, H * W]
        right_alphas = tf.concat([spatial_alphas, ch_alphas], axis=0) # [d, H * W]
        alphas = tf.concat([left_alphas, right_alphas], axis=0) # [2 * d, H * W]
        alphas = tf.reshape(alphas, shape=[2, self.dim, self.size]) # [2, d, H * W]
        alphas = tf.transpose(alphas, perm=[1, 0, 2]) # [d, 2, H * W]
        alphas = tf.reshape(alphas, shape=[2 * self.dim, self.size]) # [2 * d, H * W]

        # Method of coordinate transformation
        alpha_prods = tf.reshape(tf.gather(alphas, self.permutations), shape=[-1, self.dim, self.size])  # [2^d, d, H x W]
        idx = tf.reshape(tf.gather(interp_indices, self.permutations), shape=[-1, self.dim, self.size])  # [2^d, d, H x W]

        # Shape and size of bialteral data grid
        data_shape = tf.concat([[small_height, small_width], small_chdepths], axis=0) # [d, ]
        data_size = tf.reduce_prod(data_shape) # []

        # - Bilateral splat coordinates, shape [H x W, ]
        depths_cumprod = tf.math.cumprod(data_shape, exclusive=True, reverse=True) # [dim, ]
        splat_spcoords = tf.stack([splat_yy, splat_xx], axis=0) # [2, H, W]
        splat_coords = tf.concat([splat_spcoords, splat_chhs], axis=0) # [dim, H, W]
        splat_coords = tf.reduce_sum(depths_cumprod[:, tf.newaxis, tf.newaxis] * splat_coords, axis=0) # [H, W]
        splat_coords = tf.reshape(splat_coords, shape=[-1, ]) # [H * W, ]

        # Interpolation indices and alphas of bilateral slice
        slice_idx = tf.reshape(tf.reduce_sum(idx * depths_cumprod[tf.newaxis, :, tf.newaxis], axis=1), shape=[-1, ]) # [2^d * H * W, ]
        alpha_prod = tf.math.reduce_prod(alpha_prods, axis=1)  # [2^d, H x W]

        print("data_shape:", data_shape)

        return splat_coords, data_size, data_shape, slice_idx, alpha_prod

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # of `inp`, [H, W, N],
    #         tf.TensorSpec(shape=[None, ], dtype=tf.int32),  # of `splat_coords`, [H x W, ], int32
    #         tf.TensorSpec(shape=[], dtype=tf.int32),  # of `data_size`, [], int32
    #         tf.TensorSpec(shape=[None, ], dtype=tf.int32),  # of `data_shape`, [d, ], int32
    #         tf.TensorSpec(shape=[None, ], dtype=tf.int32),  # of `slice_idx`, [2^d x H x W, ], int32
    #         tf.TensorSpec(shape=[None, None, ], dtype=tf.float32),  # of `alpha_prod`, [H x W, ], float32
    #     ]
    # )
    def compute(
        self,
        inp: tf.Tensor,
        splat_coords: tf.Tensor,
        data_size: tf.int32,
        data_shape: tf.Tensor,
        slice_idx: tf.Tensor,
        alpha_prod: tf.Tensor,
    ) -> tf.Tensor:
        # Channel-last to channel-first because tf.map_fn works along the first axis. 
        inpT = tf.transpose(inp, perm=[2, 0, 1])

        def ch_filter(inp_ch: tf.Tensor) -> tf.Tensor:
            # Filter each channel
            inp_flat = tf.reshape(inp_ch, shape=[-1, ]) # flatten channel-wise, [H x W, ]
            # ==== Splat ====
            data_flat = tf.math.bincount(
                splat_coords,
                weights=inp_flat,
                minlength=data_size,
                maxlength=data_size,
                dtype=tf.float32,
            )
            print(data_flat.dtype)
            data = tf.reshape(data_flat, shape=data_shape)

            # ==== Blur ====
            buffer = tf.zeros_like(data)
            perm = tf.concat([tf.range(1, self.dim), [0]], axis=0) # [1, 2, 3, 4, 0] in particular for RGB

            for _ in range(self.n_iters):
                buffer, data = data, buffer

                for _ in range(self.dim):
                    # newdata = (buffer[:-2] + buffer[2:]) / 2.0
                    newdata = (buffer[:-2] + buffer[2:]) / 2.0 + buffer[1:-1] # bugfix: 1/4 x_1 + x_2 + 3/2 x_3 + x_4 + 1/4 x_5
                    data = tf.concat([data[:1], newdata, data[-1:]], axis=0)
                    data = tf.transpose(data, perm=perm)
                    buffer = tf.transpose(buffer, perm=perm)

            del buffer

            # ==== Slice ====
            data_slice = tf.gather(tf.reshape(data, shape=[-1, ]), slice_idx) # (2^dim x h x w)
            data_slice = tf.reshape(data_slice, shape=[-1, self.size])  # (2^dim, h x w)
            interpolations = alpha_prod * data_slice
            interpolation = tf.reduce_sum(interpolations, axis=0)
            interpolation = tf.reshape(interpolation, shape=[self.height, self.width])

            return interpolation

        outT = tf.map_fn(ch_filter, inpT)
        out = tf.transpose(outT, perm=[1, 2, 0])  # Channel-first to channel-last

        return out
