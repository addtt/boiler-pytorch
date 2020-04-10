from torch.utils.data import DataLoader

class BaseDatasetManager:
    """
    Wrapper for DataLoaders. Data attributes:
    - train: DataLoader object for training set
    - test: DataLoader object for test set
    - data_shape: shape of each data point (channels, height, width)
    - img_size: spatial dimensions of each data point (height, width)
    - color_ch: number of color channels
    """

    def __init__(self, cfg, cuda, **kwargs):

        # Define training and test set
        tr_set, ts_set = self.make_datasets(cfg, **kwargs)

        # Dataloaders
        self.train, self.test = self.make_dataloaders(tr_set, ts_set, cfg,
                                                      cuda, **kwargs)

        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]

    @classmethod
    def make_datasets(cls, cfg, **kwargs):
        """
        Return training and test sets as PyTorch Datasets.
        """
        raise NotImplementedError

    @classmethod
    def make_dataloaders(cls, train, test, cfg, cuda, **kwargs):
        """
        Return dataloader for training and test sets as Pytorch DataLoaders.
        Default dataloaders are provided here; override for custom dataloaders.

        :param train: (Dataset) training set
        :param test: (Dataset) test set
        :param cfg:
        :param cuda:
        :return:
        """

        # Default arguments for dataloaders
        nw = getattr(kwargs, 'num_workers', 0)
        pm = getattr(kwargs, 'pin_memory', False)
        dl_kwargs = {'num_workers': nw, 'pin_memory': pm} if cuda else {}

        dl_train = DataLoader(
            train,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            **dl_kwargs
        )
        dl_test = DataLoader(
            test,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            **dl_kwargs
        )
        return dl_train, dl_test
