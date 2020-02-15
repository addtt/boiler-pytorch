import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from boilr.utils import balanced_approx_factorization, img_grid_pad_value

img_folder = None

def _unique_filename(fname, extension):
    def exists(fname):
        return os.path.exists(fname + '.' + extension)

    if not exists(fname):
        return fname
    for i in range(10000):
        t = fname + '_' + str(i)
        if not exists(t):
            return t
    raise RuntimeError("too many files ({})".format(i))

def plot_imgs(imgs, name=None, extension='png', colorbar=True, overwrite=False):
    """
    Plots collection of 1-channel images (as 3D tensor, or 4D tensor with size
    1 on the 2nd dimension) and saves it as png. If any image extends beyond
    [0, 1], all are normalized such that the minimum and maximum are 0 and 1.

    If overwrite is False, it automatically appends an integer to the filename
    to make it unique.
    """

    if imgs.dim() == 4 and imgs.size(1) == 1:
        imgs = imgs.squeeze(1)
    if imgs.dim() != 3:
        msg = ("input tensor must be 3D, or 4D with size 1 on the 2nd "
               "dimension, but has shape {}".format(imgs.shape))
        raise RuntimeError(msg)
    if img_folder is None:
        raise RuntimeError("Image folder not set")
    fname = name.replace(' ', '_')
    fname = os.path.join(img_folder, fname)
    if not overwrite:
        fname = _unique_filename(fname, extension)
    fname = fname + '.' + extension
    imgs = imgs.detach().cpu().unsqueeze(1)   # (N, 1, H, W)
    n_imgs = imgs.size(0)
    _, c = balanced_approx_factorization(n_imgs)  # grid arrangement

    # Get minimum and maximum
    low = imgs.min().item()
    high = imgs.max().item()

    # Normalize if images extend beyond the range [0, 1]
    normalize = low < 0. or high > 1.
    if normalize:
        imgs = (imgs - low) / (high - low)

    # Compute pad value, either 0 or 1
    pad_value = img_grid_pad_value(imgs)

    # Images are now in [0, 1], arrange them into a grid as they are
    # Grid has shape (3, grid_h, grid_w) and has values in [0, 1]
    grid = make_grid(imgs, nrow=c, pad_value=pad_value, normalize=False)

    if colorbar:
        # Rescale images to original interval (now including the padding)
        grid = grid[0] * (high - low) + low

        # Save grid of images with colorbar
        plt.imshow(grid, vmin=min(low, 0.), vmax=max(high, 1.), cmap='gray')
        plt.colorbar()
        plt.title(name)
        plt.savefig(fname)
        plt.close()

    else:
        from PIL import Image

        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        grid = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
        grid = grid.permute(1, 2, 0).numpy()

        # Save image with PIL
        im = Image.fromarray(grid)
        im.save(fname, format=None)


# Test
if __name__ == '__main__':
    img_folder = ''
    low = -20.
    high = 30.
    imgs = torch.rand(4, 1, 8, 8) * (high - low) + low
    plot_imgs(imgs, name='test_colorbar_false', colorbar=False)
    plot_imgs(imgs, name='test_colorbar_true', colorbar=True)
