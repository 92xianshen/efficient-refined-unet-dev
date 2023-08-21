import tensorflow as tf

class BilateralHighDimFilterHelper(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)

        