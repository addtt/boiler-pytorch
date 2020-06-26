import argparse
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from boilr.data import BaseDatasetManager

path = 'data'


class DatasetManager(BaseDatasetManager):

    @classmethod
    def _make_datasets(cls, cfg: argparse.Namespace,
                       **kwargs) -> Tuple[Dataset, Dataset]:
        train_set = MNIST(path, train=True, download=True, transform=ToTensor())
        test_set = MNIST(path, train=False, transform=ToTensor())
        return train_set, test_set
