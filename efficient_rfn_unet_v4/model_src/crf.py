import tensorflow as tf

from .crf_config import CRFConfig


class CRF(tf.Module):
    def __init__(
        self, model_config: CRFConfig = None, computation_path: str = None, name=None
    ):
        super().__init__(name)

        self.model_config = model_config
        self.computation = tf.saved_model.load(computation_path)

    def inference(self, unary: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        return self.computation.mean_field_approximation(
            unary,
            image,
            theta_alpha=self.model_config.theta_alpha,
            theta_beta=self.model_config.theta_beta,
            theta_gamma=self.model_config.theta_gamma,
            bilateral_compat=self.model_config.bilateral_compat,
            spatial_compat=self.model_config.spatial_compat,
            compatibility=self.model_config.compatibility,
            num_iterations=self.model_config.num_iterations,
        )
