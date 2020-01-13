import numpy as np
import torch
from tqdm import tqdm

from boiler.summarize import SummarizerCollection


class BaseExperimentManager:
    """
    Experiment manager.

    Data attributes:
    - 'args': argparse.Namespace containing all config parameters. When
      initializing the MnistExperiment, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.
    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.
    - 'model': the model.
    - 'device': torch.device that is being used
    - 'dataloaders': DataLoaders, with attributes 'train' and 'test'
    - 'optimizer': the optimizer
    """

    def __init__(self, args=None):
        self.device = None
        self.dataloaders = None
        self.model = None
        self.optimizer = None
        self.args = args
        self.max_epochs = 100000   # TODO no limit for now
        if args is None:
            self.args = self._parse_args()
        self.run_description = self._make_run_description(self.args)


    @staticmethod
    def _parse_args():
        """
        Parse command-line arguments defining experiment settings.

        :return: args: argparse.Namespace with experiment settings
        """
        raise NotImplementedError


    @staticmethod
    def _make_run_description(args):
        """
        Create a string description of the run. It is used in the names of the
        logging folders.

        :param args: experiment config
        :return: the run description
        """
        raise NotImplementedError

    def make_datamanager(self):
        """
        Create a DatasetManager object. To be overridden.
        :return: DatasetManager
        """
        raise NotImplementedError

    def make_and_set_datamanager(self):
        """
        Calls the subclass's method make_datamanager() and assigns the returned
        DatasetManagers to this class's 'dataloaders' attribute.
        """
        self.dataloaders = self.make_datamanager()

    def make_model(self):
        """
        Create a model. To be overridden.
        :return: model
        """
        raise NotImplementedError

    def make_and_set_model(self):
        """
        Calls the subclass's method make_model() and assigns the returned
        model to this class's 'model' attribute.
        """
        self.model = self.make_model().to(self.device)

    def make_optimizer(self):
        """
        Create an optimizer. To be overridden.
        :return: optimizer
        """
        raise NotImplementedError

    def make_and_set_optimizer(self):
        """
        Calls the subclass's method make_optimizer() and assigns the returned
        optimizer to this class's 'optimizer' attribute.
        """
        self.optimizer = self.make_optimizer()


    def forward_pass(self, model, x, y=None):
        """
        Simple single-pass model evaluation. It consists of a forward pass
        and computation of all necessary losses and metrics.
        """
        raise NotImplementedError


    @staticmethod
    def print_train_log(step, epoch, summaries):
        raise NotImplementedError


    @staticmethod
    def print_test_log(summaries, step=None, epoch=None):
        raise NotImplementedError


    @staticmethod
    def get_metrics_dict(results):
        """
        Given a dict of results, return a dict of metrics to be given to
        summarizers. Keys are also used as names for tensorboard logging.
        """
        raise NotImplementedError


    def test_procedure(self, iw_samples=None):
        """
        Execute test procedure for the experiment. This typically includes
        collecting metrics on the test set using forward_pass().
        For example in variational inference we might be interested in
        repeating this many times to derive the importance-weighted ELBO.

        :param iw_samples: number of samples for the importance-weighted ELBO.
                The other metrics are also averaged over all these samples,
                yielding a more accurate estimate.
        :return: summaries (dict)
        """

        # Shorthand
        test_loader = self.dataloaders.test
        step = self.model.global_step
        args = self.args
        n_test = len(test_loader.dataset)

        # If it's time to estimate log likelihood, use many samples.
        # If given, use the given number.
        if iw_samples is None:
            iw_samples = 1
            if step % args.loglik_interval == 0 and step > 0:
                iw_samples = args.loglik_samples

        # Setup
        summarizers = SummarizerCollection(mode='sum')
        progress = tqdm(total=len(test_loader) * iw_samples, desc='test ')
        all_elbo_sep = torch.zeros(n_test, iw_samples)
        for batch_idx, (x, y) in enumerate(test_loader):
            for i in range(iw_samples):
                outputs = self.forward_pass(self.model, x, y)

                # elbo_sep shape (batch size,)
                i_start = batch_idx * args.test_batch_size
                i_end = (batch_idx + 1) * args.test_batch_size
                all_elbo_sep[i_start: i_end, i] = outputs['elbo_sep'].detach()

                metrics_dict = self.get_metrics_dict(outputs)
                multiplier = (x.size(0) / n_test) / iw_samples
                for k in metrics_dict:
                    metrics_dict[k] *= multiplier
                summarizers.add(metrics_dict)

                progress.update()
        progress.close()

        if iw_samples > 1:
            # Shape (test set size,)
            elbo_iw = torch.logsumexp(all_elbo_sep, dim=1)
            elbo_iw = elbo_iw - np.log(iw_samples)

            # Mean over test set (scalar)
            elbo_iw = elbo_iw.mean().item()
            key = 'elbo/elbo_IW_{}'.format(iw_samples)
            summarizers.add({key: elbo_iw})

        summaries = summarizers.get_all(reset=True)

        return summaries


    def additional_testing(self, img_folder):
        """
        Perform additional testing, including possibly generating images.

        :param img_folder: folder to store images
        """
        raise NotImplementedError
