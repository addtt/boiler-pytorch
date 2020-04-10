from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from boilr.data import BaseDatasetManager

path = 'data'

class DatasetManager(BaseDatasetManager):
    """
    Wrapper for DataLoaders. Data attributes:
    - train: DataLoader object for training set
    - test: DataLoader object for test set
    - data_shape: shape of each data point (channels, height, width)
    - img_size: spatial dimensions of each data point (height, width)
    - color_ch: number of color channels
    """

    @classmethod
    def make_datasets(cls, cfg, **kwargs):
        train_set = MNIST(path, train=True, download=True, transform=ToTensor())
        test_set = MNIST(path, train=False, transform=ToTensor())
        return train_set, test_set
