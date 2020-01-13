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

        self.cfg = cfg

        # Define training and test set
        tr_set, ts_set = self.make_datasets()

        # Dataloaders
        self.train, self.test = self.make_dataloaders(tr_set, ts_set, cuda=cuda)

        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]

    def make_datasets(self, **kwargs):
        """
        Return training and test sets as PyTorch Datasets.
        """
        raise NotImplementedError

    def make_dataloaders(self, train, test, **kwargs):
        """
        Return dataloader for training and test sets as Pytorch DataLoaders.
        Default dataloaders are provided here; override for custom dataloaders.

        :param train: (Dataset) training set
        :param test: (Dataset) test set
        :param kwargs: optional kwargs for DataLoaders
        :return:
        """

        # Default arguments for dataloaders
        nw = 0
        if 'num_workers' in kwargs:
        	nw = kwargs['num_workers']
        cuda = kwargs['cuda']
        dl_kwargs = {'num_workers': nw, 'pin_memory': False} if cuda else {}

        dl_train = DataLoader(
            train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            **dl_kwargs
        )
        dl_test = DataLoader(
            test,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            **dl_kwargs
        )
        return dl_train, dl_test
