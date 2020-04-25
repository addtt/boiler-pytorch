import os

import numpy as np
import torch
from torch import nn

from boilr.nn.utils import get_module_device
from boilr.utils import checkpoints_in_folder


class BaseModel(nn.Module):
    """Base model class."""

    def __init__(self):
        super().__init__()
        self.global_step = 0

    def increment_global_step(self):
        """Increments global step by 1."""
        self.global_step += 1

    def get_device(self):
        """Returns model device."""
        return get_module_device(self)

    def checkpoint(self, ckpt_folder, max_ckpt=None):
        """Saves model checkpoint and optionally deletes old ones.

        Args:
            ckpt_folder (str): Checkpoint folder
            max_ckpt (int, optional): If not None, maximum number of most
                recent checkpoints to be kept. Older ones are deleted.
        """

        # Get checkpoints before saving the new one (is torch.save synchronous?)
        filenames, _ = checkpoints_in_folder(ckpt_folder)

        # Save checkpoint
        path = os.path.join(ckpt_folder, "model_{}.pt".format(self.global_step))
        torch.save(self.state_dict(), path)

        # Return if we're supposed to keep all checkpoints
        if max_ckpt is None or max_ckpt == -1:
            return

        # Delete all old checkpoints except for the last max_ckpt-1
        if len(filenames) < max_ckpt - 1:
            return
        for i in range(len(filenames) - max_ckpt + 1):
            path = os.path.join(ckpt_folder, filenames[i])
            try:
                os.remove(path)
            except OSError:
                pass

    def load(self, ckpt_folder, device=None, step=None):
        """Loads model from checkpoint.

        Args:
            ckpt_folder (str): Checkpoint folder
            device (torch.device, optional): Device to move the model dict to.
                Default: None. If this is None, the model's state dict is not
                moved.
            step (int, optional): Global step from which to load the
                checkpoint. Default: None. If this is None, the latest
                checkpoint is loaded.
        """
        if step is None:
            filenames, numbers = checkpoints_in_folder(ckpt_folder)
            ckpt_name = filenames[np.argmax(numbers)]  # get latest checkpoint
            step = max(numbers)
        else:
            ckpt_name = "model_{}.pt".format(step)
        print("Loading model checkpoint at step {}...".format(step))
        path = os.path.join(ckpt_folder, ckpt_name)
        self.load_state_dict(torch.load(path, map_location=device))
        self.global_step = step
        print("Loaded.")


class BaseGenerativeModel(BaseModel):
    """Base class for generative models."""

    def sample_prior(self, n_imgs, **kwargs):
        """Samples an observation from the generative model.

        Args:
            n_imgs: Number of datapoints to be generated
        """
        raise NotImplementedError
