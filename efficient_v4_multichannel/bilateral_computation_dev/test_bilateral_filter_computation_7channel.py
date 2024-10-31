import numpy as np
import tensorflow as tf
from PIL import Image

import cv2

# from bilateral_filter_computation_multichannel import BilateralHighDimFilterComputation
from bilateral_filter_computation_ref import BilateralHighDimFilterComputation as BilateralHighDimFilterComputationRef
from bilateral_filter_computation_multichannel import BilateralHighDimFilterComputation

range_sigma = .25
space_sigma = 5.
range_padding, space_padding = 2, 2
n_iters = 2

def test_bilfilter(bil_filter, im):
    splat_coords, data_size, data_shape, slice_idx, alpha_prod = bil_filter.init(im.astype(np.float32), range_sigma=range_sigma, space_sigma=space_sigma, range_padding=range_padding, space_padding=space_padding)

    all_ones = np.ones((h, w, 1), dtype=np.float32)
    norms = bil_filter.compute(all_ones, splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod, n_iters=n_iters)
    norms = norms.numpy()

    src = im
    dst = bil_filter.compute(src.astype(np.float32), splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod, n_iters=n_iters)
    dst = dst.numpy()
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

    return dst

def test_bilfilter_multichannel(bil_filter_multichannel, im):
    ref = np.concatenate([im, im[..., 0:1]], axis=-1)
    print(ref.shape)
    splat_coords, data_size, data_shape, slice_idx, alpha_prod = bil_filter_multichannel.init(ref.astype(np.float32))

    all_ones = np.ones((h, w, 1), dtype=np.float32)
    norms = bil_filter_multichannel.compute(all_ones, splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod)
    norms = norms.numpy()

    # norm_uniq = np.unique(norms)

    # print("uniques in norm of bil filter multichannels: ", norm_uniq.shape, norm_uniq)

    src = im
    dst = bil_filter_multichannel.compute(src.astype(np.float32), splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod)
    dst = dst.numpy()
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

    return dst

# - Load data
im = Image.open("../../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

h, w, n_chs = im.shape

# bilateral_filter = BilateralHighDimFilterComputation(n_channels=n_chs)
bilateral_filter_ref = BilateralHighDimFilterComputationRef()
bilateral_filter_multichannel = BilateralHighDimFilterComputation(height=h, width=w, n_channels=4, range_sigma=range_sigma, space_sigma=space_sigma, range_padding=range_padding, space_padding=space_padding, n_iters=n_iters)

# splat_coords, data_size, data_shape, slice_idx, alpha_prod = bilateral_filter.init(im.astype(np.float32), range_sigma=.25, space_sigma=5, range_padding=2, space_padding=2)
# splat_coords2, data_size2, data_shape2, slice_idx2, alpha_prod2 = bilateral_filter_multichannel.init(im.astype(np.float32), range_sigma=.25, space_sigma=5.0, range_padding=2, space_padding=2)
# print("splat coords: ", tf.reduce_all(splat_coords == splat_coords2))
# print("data shape: ", tf.reduce_all(data_shape == data_shape2), data_shape, data_shape2)
# print("data_size: ", tf.reduce_all(data_size == data_size2))
# print("slice_idx: ", tf.reduce_all(slice_idx == slice_idx2))
# print("alpha_prod: ", tf.reduce_all(alpha_prod == alpha_prod2))
# print(data_shape)


dst1 = test_bilfilter(bilateral_filter_ref, im)
dst2 = test_bilfilter_multichannel(bilateral_filter_multichannel, im)

print("All close", np.allclose(dst1, dst2))

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst_ref", dst1[..., ::-1])
cv2.imshow("dst_multichannel", dst2[..., ::-1])
cv2.waitKey()