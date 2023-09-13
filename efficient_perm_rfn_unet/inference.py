# -*- coding: utf-8 -*-

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# - Load models
from model_src.perm_rfn_unet_x_wrapper import PermutohedralRefinedUNetX

# - Load config
from model_src.perm_rfn_unet_x_config import PermRfnUNetXConfig

# - Load utils
from util.tfrecordloader_l8 import load_testset
from util.tile_utils import reconstruct_full

# - Path to input and output
data_path = "../../data/l8/testcase/"
save_path = "output"

# - Critical hyper-parameters
theta_alpha, theta_beta = 80, 0.03125
theta_gamma = 3

# - Hyper-parameters
CROP_HEIGHT, CROP_WIDTH = 512, 512
d_bifeats, d_spfeats = 5, 2
num_bands = 7
num_classes = 4
batch_size = 1
ugenerator_path = "parameter/saved_model/unary_generator"
crf_computation_path = "parameter/saved_model/crf_computation"

config = PermRfnUNetXConfig(
    height=CROP_HEIGHT,
    width=CROP_WIDTH,
    n_channels=num_bands,
    num_classes=num_classes,
    d_bifeats=d_bifeats,
    d_spfeats=d_spfeats,
    theta_alpha=theta_alpha,
    theta_beta=theta_beta,
    theta_gamma=theta_gamma,
    bilateral_compat=10.0,
    spatial_compat=3.0,
    num_iterations=10,
)
model = PermutohedralRefinedUNetX(
    config=config,
    ugenerator_path=ugenerator_path,
)


def main():
    print("Load from {}".format(data_path))
    test_names = os.listdir(data_path)
    print("Data: {}".format(test_names))
    save_info_name = "rfn.csv"
    print("Write to {}.".format(save_info_name))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Create {}.".format(save_path))

    with open(os.path.join(save_path, save_info_name), "w") as fp:
        fp.writelines("name, theta_alpha, theta_beta, theta_gamma, duration\n")

        for test_name in test_names:
            # Names
            save_npz_name = test_name.replace("train.tfrecords", "rfn.npz")
            save_png_name = test_name.replace("train.tfrecords", "rfn.png")

            # Load one test case
            test_name = [os.path.join(data_path, test_name)]
            test_set = load_testset(test_name, batch_size=1)
            refinements = []

            # Inference
            start = time.time()
            i = 0
            for record in test_set.take(-1):
                print("Patch {}...".format(i))
                x, y = record["x_train"], record["y_train"]  # [B, H, W, C]
                # preds, rfns = inference(x)
                rfn = model(x[0])

                refinements += [rfn]

                i += 1

            refinements = np.stack(refinements, axis=0)
            refinement = reconstruct_full(
                refinements, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH
            )
            duration = time.time() - start

            # Save
            np.savez(os.path.join(save_path, save_npz_name), refinement)
            plt.imsave(os.path.join(save_path, save_png_name), refinement, cmap="gray")
            fp.writelines(
                "{}, {}, {}, {}, {}\n".format(
                    test_name, theta_alpha, theta_beta, theta_gamma, duration
                )
            )
            print("{} Done.".format(test_name))


if __name__ == "__main__":
    main()
