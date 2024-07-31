import numpy as np
import torch
def train(model, ims, real_input_flag, configs, cloud_shift, cloudless_shift):
    cost = model.train(ims, real_input_flag, cloud_shift, cloudless_shift)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag, cloud_shift, cloudless_shift)
        cost = cost / 2

    return cost