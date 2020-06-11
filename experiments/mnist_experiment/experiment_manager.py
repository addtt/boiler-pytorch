from torch import optim

import boilr
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

    def make_optimizer(self):
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

    def _parse_args(self, parser):
        """
        Parse command-line arguments defining experiment settings.

        :param: parser

        :return: args: argparse.Namespace with experiment settings
        """

        self.add_required_args(
            parser,

            # General
            batch_size=64,
            test_batch_size=1000,
            lr=1e-3,
            seed=54321,
            train_log_every=1000,
            test_log_every=1000,
            checkpoint_every=10000,
            resume="",

            # VI-specific
            loglikelihood_every=50000,
            loglikelihood_samples=100,
        )

        parser.add_argument('--wd',
                            type=float,
                            default=0.0,
                            dest='weight_decay',
                            help='weight decay')

        args = parser.parse_args()

        return args

    @staticmethod
    def _make_run_description(args):
        """
        Create a string description of the run. It is used in the names of the
        logging folders.

        :param args: experiment config
        :return: the run description
        """
        s = ''
        s += 'seed{}'.format(args.seed)
        if len(args.additional_descr) > 0:
            s += ',' + args.additional_descr
        return s
