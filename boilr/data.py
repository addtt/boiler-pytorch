from torch.utils.data import DataLoader


class BaseDatasetManager:
    """Wrapper for DataLoaders.

    This is meant to be subclassed for specific experiments and datasets.
    The method _make_datasets() must be implemented. A basic implementation of
    the method _make_dataloaders() is provided in this class, but can be
    overridden for custom behavior.

    Args:
        cfg (argparse.Namespace): Configuration
        cuda (bool): Whether the device in use is cuda
    """

    def __init__(self, cfg, cuda, **kwargs):

        # Define training and test set
        tr_set, ts_set = self._make_datasets(cfg, **kwargs)

        # Dataloaders
        self._train, self._test = self._make_dataloaders(
            tr_set, ts_set, cfg, cuda, **kwargs)

        self._data_shape = self._train.dataset[0][0].size()
        self._img_size = self._data_shape[1:]
        self._color_ch = self._data_shape[0]

    @classmethod
    def _make_datasets(cls, cfg, **kwargs):
        """Returns training and test sets as PyTorch Datasets.

        Args:
            cfg (argparse.Namespace): Configuration
        """
        raise NotImplementedError

    @classmethod
    def _make_dataloaders(cls, train, test, cfg, cuda, **kwargs):
        """Returns training and test data loaders.

        Default data loaders provided here. Override for custom data loaders.

        Args:
            train (Dataset): Training set
            test (Dataset): Test set
            cfg (argparse.Namespace): Configuration
            cuda (bool): Whether the device in use is cuda

        Returns:
            (tuple): tuple containing:
                - dl_train (DataLoader): training set data loader
                - dl_test (DataLoader): test set data loader
        """

        # Default arguments for dataloaders
        nw = getattr(kwargs, 'num_workers', 0)
        pm = getattr(kwargs, 'pin_memory', False)
        dl_kwargs = {'num_workers': nw, 'pin_memory': pm} if cuda else {}

        dl_train = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              drop_last=True,
                              **dl_kwargs)
        dl_test = DataLoader(test,
                             batch_size=cfg.test_batch_size,
                             shuffle=False,
                             **dl_kwargs)
        return dl_train, dl_test

    @property
    def train(self):
        """DataLoader for training set"""
        return self._train

    @property
    def test(self):
        """DataLoader for test set"""
        return self._test

    @property
    def data_shape(self):
        """Shape of each datapoint (image): (channels, height, width)."""
        return self._data_shape

    @property
    def img_size(self):
        """Spatial shape of each datapoint (image): height, width."""
        return self._img_size

    @property
    def color_ch(self):
        """Number of color channels of each datapoint (image)."""
        return self._color_ch
