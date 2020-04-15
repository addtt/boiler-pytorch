import datetime
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
from .options import get_options

__sentinel = object()

def print_num_params(model, max_depth=__sentinel):

    if max_depth is __sentinel:
        max_depth = get_options('model_print_depth')
    assert max_depth is None or isinstance(max_depth, int)

    sep = '.'  # string separator in parameter name
    print("\n--- Trainable parameters:")
    num_params_tot = 0
    num_params_dict = OrderedDict()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()

        if max_depth is not None:
            split = name.split(sep)
            prefix = sep.join(split[:max_depth])
        else:
            prefix = name
        if prefix not in num_params_dict:
            num_params_dict[prefix] = 0
        num_params_dict[prefix] += num_params
        num_params_tot += num_params
    for n, n_par in num_params_dict.items():
        print("{:8d}  {}".format(n_par, n))
    print("  - Total trainable parameters:", num_params_tot)
    print("---------\n")


def set_rnd_seed(seed, aggressive=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # The two lines below might slow down training
    if aggressive:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_date_str():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def linear_anneal(x, start, end, steps):
    assert x >= 0
    assert steps > 0
    assert start >= 0
    assert end >= 0
    if x > steps:
        return end
    if x < 0:
        return start
    return start + (end - start) / steps * x


def to_np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def get_module_device(module):
    return next(module.parameters()).device


def is_conv(m):
    return isinstance(m, torch.nn.modules.conv._ConvNd)


def is_linear(m):
    return isinstance(m, torch.nn.Linear)


def named_leaf_modules(module):
    # Should work under common naming assumptions, but it's not guaranteed
    last_name = ''
    for name, l in reversed(list(module.named_modules())):
        if name not in last_name:
            last_name = name
            yield name, l
