import datetime
import os
import random
import re

import numpy as np
import torch


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


def checkpoints_in_folder(folder):
    """Finds checkpoints in a folder.

    Finds checkpoint files in the speficied folder, return a list of file names
    and a list of integers corresponding to the time steps when the checkpoints
    were saved. These lists have the same length, they have a one-to-one
    correspondence, and they are sorted by number.

    Arguments:
        folder (str)

    Returns:
        filenames (list of str): File names of checkpoints. The full path is
            obtained by joining the input argument folder with these file
            names.
        numbers (list of int)
    """

    def is_checkpoint_file(f):
        # TODO use regex properly and maybe get number directly here
        full_path = os.path.join(folder, f)
        return (os.path.isfile(full_path) and f.startswith("model_") and
                f.endswith('.pt'))

    filenames = [f for f in os.listdir(folder) if is_checkpoint_file(f)]
    regex = re.compile(r'\d+')
    numbers = list([int(regex.search(n).group(0)) for n in filenames])
    assert len(filenames) == len(numbers)
    filenames = list([filenames[i] for i in np.argsort(numbers)])
    return filenames, numbers
