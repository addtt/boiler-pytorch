"""
Useful basic ops for neural nets. Mostly simple wrappers.
"""

from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    def __init__(self, size=None, scale=None, mode='bilinear', align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners
        )
        return out


class CropImage(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


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
    """
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: iterable (height, width)
    :return: padded image
    """
    return _pad_crop_img(x, size, 'pad')


def crop_img_tensor(x, size):
    """
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: iterable (height, width)
    :return: cropped image
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x, size, mode):
    """
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple size.
    :param x: input image (tensor)
    :param size: tuple (height, width)
    :param mode: string ('pad' | 'crop')
    :return: padded image
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
