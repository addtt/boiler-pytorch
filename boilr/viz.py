import os

from torchvision.utils import save_image

from .utils import balanced_approx_factorization, img_grid_pad_value

img_folder = None


def maybe_rescale_intensities(imgs):
    mn = imgs.min()
    mx = imgs.max()
    if mn < 0.0 or mx > 1.0:
        imgs = (imgs - mn) / (mx - mn)
    return imgs


def plot_imgs(imgs, name=None):
    """
    Plots collection of 1-channel images (as 3D tensor, or 4D tensor with size
    1 on the 2nd dimension) and saves it as png. If any image extends beyond
    [0, 1], all are normalized such that the minimum and maximum are 0 and 1.
    """

    if imgs.dim() == 4 and imgs.size(1) == 1:
        imgs = imgs.squeeze(1)
    if imgs.dim() != 3:
        msg = ("input tensor must be 3D, or 4D with size 1 on the 2nd "
               "dimension, but has shape {}".format(imgs.shape))
        raise RuntimeError(msg)
    if img_folder is None:
        raise RuntimeError("Image folder not set")
    imgs = imgs.detach().cpu()
    fname = name.replace(' ', '_') + '.png'
    fname = os.path.join(img_folder, fname)
    n = imgs.size(0)
    _, c = balanced_approx_factorization(n)
    imgs = maybe_rescale_intensities(imgs)
    imgs = imgs.unsqueeze(1)
    pad_value = img_grid_pad_value(imgs)
    save_image(imgs, fname, nrow=c, pad_value=pad_value)
