import datetime
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data


def print_num_params(model, max_depth=None):
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
        print("{:7d}  {}".format(n_par, n))
    print("  - Total trainable parameters:", num_params_tot)
    print("---------\n")


def set_rnd_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # The two lines below might slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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


############################
### VIZ

def img_grid_pad_value(imgs, thresh=.2):
    """
    Hack to visualize boundaries between images with torchvision's save_image().
    If the median border value of all images is below the threshold, use white,
    otherwise black (which is the default)
    :param imgs: 4d tensor
    :param thresh: threshold in (0, 1)
    :return: padding value
    """

    assert imgs.dim() == 4
    imgs = imgs.clamp(min=0., max=1.)
    assert 0. < thresh < 1.

    imgs = imgs.mean(1)  # reduce to 1 channel
    h = imgs.size(1)
    w = imgs.size(2)
    borders = list()
    borders.append(imgs[:, 0].flatten())
    borders.append(imgs[:, h - 1].flatten())
    borders.append(imgs[:, 1:h - 1, 0].flatten())
    borders.append(imgs[:, 1:h - 1, w - 1].flatten())
    borders = torch.cat(borders)
    if torch.median(borders) < thresh:
        return 1.0
    return 0.0


def balanced_approx_factorization(x, ratio=1):
    """
    :param x:
    :param ratio: ratio columns/rows
    :return:
    """

    assert type(x) == int or x.dtype == int

    # We want c/r to be approx equal to ratio, and r*c to be approx equal to x
    # ==> r = x/c = x/(ratio*r)
    # ==> r = sqrt(x/ratio) and c = sqrt(x*ratio)
    c = np.ceil(np.sqrt(x * ratio))
    r = np.ceil(x / c)
    return r, c



def clean_axes():
    plt.gca().tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks
        left=False,  # ticks along the left edge are off
        right=False,
        bottom=False,
        top=False,
        labelleft=False,  # labels along the left edge are off
        labelbottom=False)


############################
# Saving model activations


def save_activation_hook(ord_dict, name):
    def hook(model, inp, ret):
        if isinstance(ret, tuple):
            try:
                ret = torch.cat(ret, dim=1)
            except TypeError:
                ret = ret[0]
            except RuntimeError as e:
                print("WARNING:", e)
                return
        ord_dict[name] = ret.detach()
    return hook


def set_up_saving_all_activations(model):
    all_activations = OrderedDict()
    for module_name, module in named_leaf_modules(model):
        module.register_forward_hook(save_activation_hook(all_activations, module_name))
    return all_activations


if __name__ == '__main__':
    # Test
    img_grid_pad_value(torch.rand(6, 3, 32, 32), thresh=.3)
