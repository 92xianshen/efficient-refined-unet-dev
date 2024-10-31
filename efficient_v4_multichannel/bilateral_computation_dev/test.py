import tensorflow as tf

range_sigma = 0.03125
space_sigma = 80
range_padding = 2
space_padding = 2

# def clamp(
#     x: tf.Tensor, min_value: tf.float32, max_value: tf.float32
# ) -> tf.Tensor:
#     return tf.maximum(min_value, tf.minimum(max_value, x))

# def get_both_indices(coord: tf.Tensor, size: tf.int32) -> tf.Tensor:
#     """
#     Method to get left and right indices of slice interpolation.
#     """
#     left_index = clamp(
#         tf.cast(coord, dtype=tf.int32), min_value=0, max_value=size - 1
#     )
#     right_index = clamp(left_index + 1, min_value=0, max_value=size - 1)
#     return left_index, right_index

def get_both_indices(coord: tf.Tensor, size: tf.Tensor) -> tf.Tensor:
    """
    Method to get left and right indices of slice interpolation.
    """
    left_indices = tf.maximum(0, tf.minimum(size[tf.newaxis, tf.newaxis, :] - 1, tf.cast(coord, dtype=tf.int32)))
    right_indices = tf.maximum(0, tf.minimum(size[tf.newaxis, tf.newaxis, :] - 1, left_indices + 1))
    return left_indices, right_indices

# Decompose `features` into r, g, and b channels
def create_range_params(ch):
    ch_min, ch_max = tf.reduce_min(ch), tf.reduce_max(ch)
    ch_delta = ch_max - ch_min
    # - Range coordinates, shape [H, W], dtype float
    chh = ch - ch_min

    # - Depths of data grid
    small_chdepth = (
        tf.cast(ch_delta / range_sigma, dtype=tf.int32) + 1 + 2 * range_padding
    )

    # - Range coordinates of splat, shape [H, W]
    splat_chh = tf.cast(chh / range_sigma + 0.5, dtype=tf.int32) + range_padding

    # - Range coordinates of slice, shape [H, W]
    slice_chh = chh / range_sigma + tf.cast(range_padding, dtype=tf.float32)

    # - Slice interpolation range coordinate pairs
    ch_index, chh_index = get_both_indices(
        slice_chh, small_chdepth
    )  # [H, W], [H, W]

    # - Intepolation factors
    ch_alpha = tf.reshape(
        slice_chh - tf.cast(ch_index, dtype=tf.float32),
        shape=[
            -1,
        ],
    )  # [H x W, ]

    return small_chdepth, splat_chh, ch_index, chh_index, ch_alpha

def create_range_params_multichannel(image):
    n_channels = tf.shape(image)[-1]
    ch_mins, ch_maxs = tf.reduce_min(image, axis=[0, 1]), tf.reduce_max(image, axis=[0, 1]) # [C, ], [C, ], in particular [3, ]
    ch_deltas = ch_maxs - ch_mins # [C, ]
    # - Range coordinates, shape [H, W], dtype float
    chhs = image - ch_mins[tf.newaxis, tf.newaxis] # [H, W, C]

    # - Depths of data grid
    small_chdepths = (tf.cast(ch_deltas / range_sigma, dtype=tf.int32) + 1 + 2 * range_padding) # [C, ]

    # - Range coordinates of splat, shape [H, W]
    splat_chhs = tf.cast(chhs / range_sigma + 0.5, dtype=tf.int32) + range_padding # [H, W, C]

    # - Range coordinates of slice, shape [H, W]
    slice_chhs = chhs / range_sigma + tf.cast(range_padding, dtype=tf.float32) # [H, W, C]

    # - Slice interpolation range coordinate pairs
    # ch_indices, chh_indices = get_both_indices(
    #     slice_chhs, small_chdepths
    # )  

    # - Slice interpolation range coordinate pairs
    ch_indices = tf.maximum(0, tf.minimum(small_chdepths[tf.newaxis, tf.newaxis, :] - 1, tf.cast(slice_chhs, dtype=tf.int32))) # [H, W, C]
    chh_indices = tf.maximum(0, tf.minimum(small_chdepths[tf.newaxis, tf.newaxis, :] - 1, ch_indices + 1)) # [H, W, C]

    # - Intepolation factors
    ch_alphas = tf.reshape(
        slice_chhs - tf.cast(ch_indices, dtype=tf.float32),
        shape=[
            -1, n_channels
        ],
    )  # [H x W, ]

    return small_chdepths, splat_chhs, ch_indices, chh_indices, ch_alphas

if __name__ == "__main__":
    image = tf.ones([512, 512, 3])
    # imageT = tf.transpose(image, perm=[2, 0, 1])
    small_chdepths, splat_chhs, ch_indices, chh_indices, ch_alphas = create_range_params(image)
    print(small_chdepths.shape, splat_chhs.shape)