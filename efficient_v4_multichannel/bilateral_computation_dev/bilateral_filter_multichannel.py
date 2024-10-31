
import tensorflow as tf

from .bilateral_filter_computation_multichannel import BilateralHighDimFilterComputation

class BilateralFilterMultichannel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.computation = BilateralHighDimFilterComputation()
        self.

    def init(self, image, range_sigma, space_sigma, range_padding, space_padding):
        splat_coords, data_size, data_shape, slice_idx, alpha_prod = self.computation.init(image, range_sigma, space_sigma, range_padding, space_padding)