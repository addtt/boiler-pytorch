import datetime
import math
import os
import random
import re
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


def grad_norm(parameters, norm_type=2):
    r"""Compute gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    This code is based on torch.nn.utils.clip_grad_norm_(), with minor
    modifications.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


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


def checkpoints_in_folder(folder):
    r"""Find checkpoints in a folder.

    Find checkpoint files in the speficied folder, return a list of file names
    and a list of integers corresponding to the time steps when the checkpoints
    were saved. These lists have the same length, they have a one-to-one
    correspondence, and they are sorted by number.

    Arguments:
        folder (str)

    Returns:
        filenames (list of str): File names of checkpoints. The full path is
        obtained by joining the input argument folder with these file names.
        numbers (list of int)
    """

    def is_checkpoint_file(f):
        # TODO use regex properly and maybe get number directly here
        full_path = os.path.join(folder, f)
        return (os.path.isfile(full_path) and
                f.startswith("model_") and
                f.endswith('.pt'))

    filenames = [f for f in os.listdir(folder) if is_checkpoint_file(f)]
    regex = re.compile(r'\d+')
    numbers = list([int(regex.search(n).group(0)) for n in filenames])
    assert len(filenames) == len(numbers)
    filenames = list([filenames[i] for i in np.argsort(numbers)])
    return filenames, numbers
