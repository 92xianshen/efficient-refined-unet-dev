import tensorflow as tf


class SpatialHighDimFilterConfig(tf.Module):
    """
    Config class of bilateral high-dim filter
    """

    def __init__(
        self,
        space_sigma: float = 16,
        space_padding: int = 2,
        n_iters: int = 2, 
        name: str = None,
    ):
        super().__init__(name)

        self.space_sigma = tf.constant(space_sigma, dtype=tf.float32)
        self.space_padding = tf.constant(space_padding, dtype=tf.int32)

        self.n_iters = tf.constant(n_iters, dtype=tf.int32)
