import tensorflow as tf


class BilateralHighDimFilterConfig(tf.Module):
    """
    Config class of bilateral high-dim filter
    """

    def __init__(
        self,
        range_sigma: float = 0.25,
        space_sigma: float = 16,
        range_padding: int = 2,
        space_padding: int = 2,
        n_iters: int = 2, 
        name: str = None,
    ):
        super().__init__(name)

        self.range_sigma = tf.constant(range_sigma, dtype=tf.float32)
        self.space_sigma = tf.constant(space_sigma, dtype=tf.float32)
        self.range_padding = tf.constant(range_padding, dtype=tf.int32)
        self.space_padding = tf.constant(space_padding, dtype=tf.int32)

        self.n_iters = tf.constant(n_iters, dtype=tf.int32)
