from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from boilr.data import BaseDatasetManager

path = 'data'


class DatasetManager(BaseDatasetManager):

    @classmethod
    def _make_datasets(cls, cfg, **kwargs):
        train_set = MNIST(path, train=True, download=True, transform=ToTensor())
        test_set = MNIST(path, train=False, transform=ToTensor())
        return train_set, test_set
