import tensorflow as tf

class CRFConfig(tf.Module):
    def __init__(self, theta_alpha: float=80.0, theta_beta: float=0.0625, theta_gamma: float=3.0, bilateral_compat: float=10.0, spatial_compat: float=3.0, num_iterations: int=10, name=None):
        super().__init__(name)

        self.theta_alpha, self.theta_beta = tf.constant(theta_alpha, dtype=tf.float32), tf.constant(theta_beta, dtype=tf.float32)
        self.theta_gamma = tf.constant(theta_gamma, dtype=tf.float32)

        self.bilateral_compat = tf.constant(bilateral_compat, dtype=tf.float32)
        self.spatial_compat = tf.constant(spatial_compat, dtype=tf.float32)
        self.compatibility = tf.constant(-1, dtype=tf.float32)
        self.num_iterations = tf.constant(num_iterations, dtype=tf.int32)
