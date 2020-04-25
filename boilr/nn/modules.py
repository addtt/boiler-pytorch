"""Useful basic layers and ops for neural nets."""

from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self,
                 size=None,
                 scale=None,
                 mode='bilinear',
                 align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(x,
                            size=self.size,
                            scale_factor=self.scale,
                            mode=self.mode,
                            align_corners=self.align_corners)
        return out


class CropImage(nn.Module):
    """Crops image to given size.

    Args:
        size
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size)


class Reshape(nn.Module):

    def __init__(self, *args, implicit_batch=True, allow_copy=False):
        super().__init__()
        self.shape = args
        self.implicit_batch = implicit_batch
        self.allow_copy = allow_copy

    def forward(self, x):
        shp = self.shape
        if self.implicit_batch:
            shp = (x.size(0), *shp)
        if self.allow_copy:
            return x.reshape(shp)
        return x.view(shp)


class PrintShape(nn.Module):

    def __init__(self):
        super().__init__()
        self.first_pass = True

    def forward(self, x):
        if self.first_pass:
            print(" > shape:", x.shape)
            self.first_pass = False
        return x


class Identity(nn.Module):

    def forward(self, x):
        return x


def pad_img_tensor(x, size):
    """Pads a tensor.

    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.

    Args:
        x (torch.Tensor): Input image
        size (iterable): Desired size (height, width)

    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')


def crop_img_tensor(x, size):
    """Crops a tensor.

    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.

    Args:
        x (torch.Tensor): Input image
        size (iterable): Desired size (height, width)

    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x, size, mode):
    """ Pads or crops a tensor.

    Pads or crops a tensor of shape (batch, channels, h, w) to new height
    and width given by a tuple.

    Args:
        x (torch.Tensor): Input image
        size (iterable): Desired size (height, width)
        mode (str): Mode, either 'pad' or 'crop'

    Returns:
        The padded or cropped tensor
    """

    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]


def free_bits_kl(kl, free_bits, batch_average=False, eps=1e-6):
    """Computes free-bits version of KL divergence.

    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.

    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).

    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)

    Returns:
        The KL with free bits
    """

    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)
