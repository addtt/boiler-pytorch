from os.path import join

import numpy as np
import torch
from torch import nn

from .utils import get_module_device, checkpoints_in_folder
import os

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_step = 0

    def increment_global_step(self):
        self.global_step += 1

    def get_device(self):
        return get_module_device(self)

    def checkpoint(self, ckpt_folder, max_ckpt=None):
        # Get checkpoints before saving the new one (is torch.save synchronous?)
        filenames, _ = checkpoints_in_folder(ckpt_folder)

        # Save checkpoint
        path = join(ckpt_folder, "model_{}.pt".format(self.global_step))
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
        if step is None:
            filenames, numbers = checkpoints_in_folder(ckpt_folder)
            ckpt_name = filenames[np.argmax(numbers)]   # get latest checkpoint
            step = max(numbers)
        else:
            ckpt_name = "model_{}.pt".format(step)
        print("Loading model checkpoint at step {}...".format(step))
        path = join(ckpt_folder, ckpt_name)
        self.load_state_dict(torch.load(path, map_location=device))
        self.global_step = step
        print("Loaded.")


class BaseGenerativeModel(BaseModel):
    def sample_prior(self, n_imgs, **kwargs):
        raise NotImplementedError
