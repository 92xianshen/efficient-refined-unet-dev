"""
Bilateral high-dim filter, built with TF 2.x.
"""

import tensorflow as tf

from .bilateral_filter_config import BilateralHighDimFilterConfig


class BilateralHighDimFilter(tf.Module):
    def __init__(
        self,
        model_config: BilateralHighDimFilterConfig = None,
        computation_path: str = None,
        name: str = None,
    ) -> None:
        super().__init__(name)

        self.model_config = model_config
        self.computation = tf.saved_model.load(computation_path)

        (
            self.splat_coords,
            self.data_size,
            self.data_shape,
            self.slice_idx,
            self.alpha_prod,
        ) = (None, None, None, None, None)

    def init(self, features: tf.Tensor = None) -> None:
        (
            self.splat_coords,
            self.data_size,
            self.data_shape,
            self.slice_idx,
            self.alpha_prod,
        ) = self.computation.init(
            features,
            range_sigma=self.model_config.range_sigma,
            space_sigma=self.model_config.space_sigma,
            range_padding=self.model_config.range_padding,
            space_padding=self.model_config.space_padding,
        )

    def compute(self, inp: tf.Tensor = None) -> None:
        return self.computation.compute(
            inp,
            splat_coords=self.splat_coords,
            data_size=self.data_size,
            data_shape=self.data_shape,
            slice_idx=self.slice_idx,
            alpha_prod=self.alpha_prod,
            n_iters=self.model_config.n_iters,
        )
