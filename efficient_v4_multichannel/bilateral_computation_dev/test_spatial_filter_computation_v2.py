import numpy as np
import tensorflow as tf
from PIL import Image

import cv2

from spatial_filter_computation_v2 import SpatialHighDimFilterComputation
from spatial_filter_computation_ref import SpatialHighDimFilterComputation as SpatialHighDimFilterComputationRef

# - Load data
im = Image.open("../../../data/lenna.png").convert("RGB")
im = np.array(im) / 255.0

space_sigma = 3.

h, w, n_chs = im.shape

# def test_bilfilter(bil_filter, im):
#     splat_coords, data_size, data_shape, slice_idx, alpha_prod = bil_filter.init(im.astype(np.float32), range_sigma=.25, space_sigma=5.0, range_padding=2, space_padding=2)

#     all_ones = np.ones((h, w, 1), dtype=np.float32)
#     norms = bil_filter.compute(all_ones, splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod, n_iters=2)
#     norms = norms.numpy()

#     src = im
#     dst = bil_filter.compute(src.astype(np.float32), splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod, n_iters=2)
#     dst = dst.numpy()
#     dst = dst / norms
#     dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

#     return dst

# def test_bilfilter_multichannel(bil_filter_multichannel, im):
#     splat_coords, data_size, data_shape, slice_idx, alpha_prod = bil_filter_multichannel.init(im.astype(np.float32))

#     all_ones = np.ones((h, w, 1), dtype=np.float32)
#     norms = bil_filter_multichannel.compute(all_ones, splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod)
#     norms = norms.numpy()

#     src = im
#     dst = bil_filter_multichannel.compute(src.astype(np.float32), splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod)
#     dst = dst.numpy()
#     dst = dst / norms
#     dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

#     return dst

def test_spfilter_ref(sp_filter_ref, im):
    splat_coords, data_size, data_shape, slice_idx, alpha_prod = sp_filter_ref.init(height=h, width=w, space_sigma=space_sigma, space_padding=2)

    all_ones = np.ones((h, w, 1), dtype=np.float32)
    norms = sp_filter_ref.compute(all_ones, splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod, n_iters=2)
    norms = norms.numpy()

    src = im
    dst = sp_filter_ref.compute(src.astype(np.float32), splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod, n_iters=2)
    dst = dst.numpy()
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

    return dst

def test_spfilter_v2(sp_filter_v2, im):
    splat_coords, data_size, data_shape, slice_idx, alpha_prod = sp_filter_v2.init()

    all_ones = np.ones((h, w, 1), dtype=np.float32)
    norms = sp_filter_v2.compute(all_ones, splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod)
    norms = norms.numpy()

    src = im
    dst = sp_filter_v2.compute(src.astype(np.float32), splat_coords=splat_coords, data_size=data_size, data_shape=data_shape, slice_idx=slice_idx, alpha_prod=alpha_prod)
    dst = dst.numpy()
    dst = dst / norms
    dst = (dst - dst.min()) / (dst.max() - dst.min() + 1e-5)

    return dst
# # bilateral_filter = BilateralHighDimFilterComputation(n_channels=n_chs)
# bilateral_filter_ref = BilateralHighDimFilterComputationRef()
# bilateral_filter_multichannel = BilateralHighDimFilterComputation(height=h, width=w, n_channels=n_chs, range_sigma=.25, space_sigma=5.0, range_padding=2, space_padding=2, n_iters=2)

# splat_coords, data_size, data_shape, slice_idx, alpha_prod = bilateral_filter.init(im.astype(np.float32), range_sigma=.25, space_sigma=5, range_padding=2, space_padding=2)
# splat_coords2, data_size2, data_shape2, slice_idx2, alpha_prod2 = bilateral_filter_multichannel.init(im.astype(np.float32), range_sigma=.25, space_sigma=5.0, range_padding=2, space_padding=2)
# print("splat coords: ", tf.reduce_all(splat_coords == splat_coords2))
# print("data shape: ", tf.reduce_all(data_shape == data_shape2), data_shape, data_shape2)
# print("data_size: ", tf.reduce_all(data_size == data_size2))
# print("slice_idx: ", tf.reduce_all(slice_idx == slice_idx2))
# print("alpha_prod: ", tf.reduce_all(alpha_prod == alpha_prod2))
# print(data_shape)


# dst1 = test_bilfilter(bilateral_filter_ref, im)
# dst2 = test_bilfilter_multichannel(bilateral_filter_multichannel, im)

# print("All close", np.allclose(dst1, dst2))

sp_filter_v2 = SpatialHighDimFilterComputation(height=h, width=w, space_sigma=space_sigma, space_padding=2, n_iters=2)
sp_filter_ref = SpatialHighDimFilterComputationRef()
dst = test_spfilter_v2(sp_filter_v2, im)
dst_ref = test_spfilter_ref(sp_filter_ref, im)

dst_cv = cv2.GaussianBlur(im.astype(np.float32), (0, 0), space_sigma)

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst_ref", dst_ref[..., ::-1])
# cv2.imshow("dst_multichannel", dst2[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.imshow("dst_cv", dst_cv[..., ::-1])
cv2.waitKey()