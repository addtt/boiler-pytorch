import argparse
from typing import Optional

from torch import optim

import boilr.data
from boilr import VAEExperimentManager
from models.mnist_vae import MnistVAE
from .data import DatasetManager

boilr.set_options(model_print_depth=2)
# boilr.set_options(show_progress_bar=False)


class MnistExperiment(VAEExperimentManager):
    """
    Experiment manager.

    Data attributes:
    - 'args': argparse.Namespace containing all config parameters. When
      initializing this object, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.
    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.
    - 'model': the model.
    - 'device': torch.device that is being used
    - 'dataloaders': DataLoaders, with attributes 'train' and 'test'
    - 'optimizer': the optimizer
    """

    def _make_datamanager(self):
        cuda = self.device.type == 'cuda'
        return DatasetManager(self.args, cuda)

    def _make_model(self):
        return MnistVAE()

    def _make_optimizer(self):
        return optim.Adam(self.model.parameters(),
                          lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

    def forward_pass(self, x, y=None):
        """
        Simple single-pass model evaluation. It consists of a forward pass
        and computation of all necessary losses and metrics.
        """

        x = x.to(self.device, non_blocking=True)
        out = self.model(x)
        elbo_sep = out['elbo']
        elbo = elbo_sep.mean()
        loss = -elbo

        out = {
            'out_sample': out['sample'],
            'out_mean': out['mean'],
            'loss': loss,
            'elbo_sep': elbo_sep,
            'elbo/elbo': elbo,
            'elbo/recons': out['nll'].mean(),
            'elbo/kl': out['kl'].mean(),
        }
        return out

    @classmethod
    def _define_args_defaults(cls) -> dict:
        defaults = super(MnistExperiment, cls)._define_args_defaults()
        defaults.update(
            # General
            batch_size=64,
            test_batch_size=1000,
            lr=1e-3,
            seed=54321,
            train_log_every=1000,
            test_log_every=1000,
            checkpoint_every=1000,
            keep_checkpoint_max=3,
            resume="",

            # VI-specific
            loglikelihood_every=50000,
            loglikelihood_samples=100,
        )
        return defaults

    def _add_args(self, parser: argparse.ArgumentParser) -> None:

        super(MnistExperiment, self)._add_args(parser)

        parser.add_argument('--wd',
                            type=float,
                            default=0.0,
                            dest='weight_decay',
                            help='weight decay')

    @classmethod
    def _check_args(cls, args: argparse.Namespace) -> argparse.Namespace:
        args = super(MnistExperiment, cls)._check_args(args)
        if args.weight_decay < 0:
            raise ValueError("'weight_decay' must be nonnegative")
        return args

    @staticmethod
    def _make_run_description(args):
        # Can be (but doesn't have to be) overridden
        s = ''
        s += 'seed{}'.format(args.seed)
        if len(args.additional_descr) > 0:
            s += ',' + args.additional_descr
        return s

    def post_backward_callback(self) -> None:
        # Can be (but doesn't have to be) overridden
        super(MnistExperiment, self).post_backward_callback()

    @classmethod
    def get_metrics_dict(cls, results: dict) -> dict:
        # Can be (but doesn't have to be) overridden
        return super(MnistExperiment, cls).get_metrics_dict(results)

    @classmethod
    def train_log_str(cls,
                      summaries: dict,
                      step: int,
                      epoch: Optional[int] = None) -> str:
        # Can be (but doesn't have to be) overridden
        return super(MnistExperiment, cls).train_log_str(summaries,
                                                         step,
                                                         epoch=epoch)

    @classmethod
    def test_log_str(cls,
                     summaries: dict,
                     step: int,
                     epoch: Optional[int] = None) -> str:
        # Can be (but doesn't have to be) overridden
        return super(MnistExperiment, cls).test_log_str(summaries,
                                                        step,
                                                        epoch=epoch)
