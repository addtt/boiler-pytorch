import argparse
import os
import pickle
import warnings
from typing import Type, Optional

import torch

from boilr.experiments import BaseExperimentManager
from boilr.utils import viz
from boilr.utils.meta import ObjectWithArgparsedArgs
from boilr.utils.utils import get_date_str


class BaseOfflineEvaluator(ObjectWithArgparsedArgs):
    """Boilerplate code to run evaluation routines on a trained model.

    Initialize with the experiment class used for training (the class, not the
    instance).

    A subclass must define:
    - the _parse_args() method, which is called during init, and must include a
      call to add_required_args() with the argparser as argument.
    - the run() method, defining the whole evaluation procedure.

    The evaluator can then be called directly because it implements __call__().

    The following data attributes are defined, to be used by subclasses:
    - self._experiment
    - self._img_folder

    Args:
        args (argparse.Namespace, optional): Unused.
        experiment_class (class): Class object of (a subclass of)
            `BaseExperimentManager`.
    """

    def __init__(self,
                 args: Optional[argparse.Namespace] = None,
                 experiment_class: Type[BaseExperimentManager] = None):
        del args
        if experiment_class is None:
            raise ValueError("Argument `experiment_class` cannot be None")
        super(BaseOfflineEvaluator, self).__init__()

        args = self.args  # these are the evaluation args
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        date_str = get_date_str()
        print('device: {}, start time: {}'.format(device, date_str))

        # Get path to load model
        checkpoint_folder = os.path.join('output', args.load, 'checkpoints')

        # Add date string and create folder on evaluation_results
        self._result_folder = os.path.join('output', args.load,
                                           'evaluation_output',
                                           'eval_' + date_str)
        self._img_folder = os.path.join(self._result_folder, 'imgs')
        os.makedirs(self._result_folder)
        os.makedirs(self._img_folder)

        # Set img folder for viz module
        viz.img_folder = self._img_folder

        # Load config (this contains the *original* experiment args)
        config_path = os.path.join(checkpoint_folder, 'config.pkl')
        with open(config_path, 'rb') as file:
            experiment_args = pickle.load(file)

        # Modify experiment config for testing
        if args.test_batch_size is not None:
            experiment_args.test_batch_size = args.test_batch_size
        experiment_args.dry_run = False

        # Create and set up experiment manager
        experiment = experiment_class(args=experiment_args)
        experiment.setup(device=device, checkpoint_folder=checkpoint_folder)

        self._experiment = experiment

    @classmethod
    def _define_args_defaults(cls) -> dict:
        defaults = super(BaseOfflineEvaluator, cls)._define_args_defaults()
        defaults.update(
            default_run="",
            test_batch_size=None,
        )
        return defaults

    def _add_args(self, parser: argparse.ArgumentParser) -> None:
        super(BaseOfflineEvaluator, self)._add_args(parser)

        parser.add_argument('--load',
                            type=str,
                            metavar='NAME',
                            default=self._default_args['default_run'],
                            help="name of the run to be loaded")
        parser.add_argument('--load-step',
                            type=int,
                            dest='load_step',
                            metavar='N',
                            help='step of checkpoint to be loaded (default:'
                            ' last available)')
        parser.add_argument('--test-batch-size',
                            type=int,
                            default=self._default_args['test_batch_size'],
                            dest='test_batch_size',
                            metavar='N',
                            help='test batch size')
        parser.add_argument('--nocuda',
                            action='store_true',
                            dest='no_cuda',
                            help='do not use cuda')

    @classmethod
    def _check_args(cls, args: argparse.Namespace) -> None:
        super(BaseOfflineEvaluator, cls)._check_args(args)
        if args.load_step is not None:
            warnings.warn(
                "Loading weights from specific training step is not supported "
                "for now. The model will be loaded from the last checkpoint.")

    def run(self):
        """Runs the evaluator."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Runs the evaluator."""
        self.run()
