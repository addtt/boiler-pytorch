import os
import pickle
import warnings

import torch

from boilr.utils import viz
from boilr.utils.utils import get_date_str


class BaseOfflineEvaluator:  # TODO test this in example.py
    """Boilerplate code to run evaluation routines on a trained model.

    Initialize with the experiment class used for training (the class, not the
    instance), and optionally with the string description of a run that is
    going to be the default run to load from.

    A subclass must define:
    - the _parse_args() method, which is called during init, and must include a
      call to add_required_args() with the argparser as argument.
    - the run() method, defining the whole evaluation procedure.

    The evaluator can then be called directly because it implements __call__().

    The following data attributes are defined, to be used by subclasses:
    - self._experiment
    - self._img_folder
    - self._eval_args

    Args:
        experiment_class (class)
        default_run (str, optional): default run to load the model from
    """

    def __init__(self, experiment_class, default_run=""):
        self._default_run = default_run
        eval_args = self._parse_args()

        use_cuda = not eval_args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        date_str = get_date_str()
        print('device: {}, start time: {}'.format(device, date_str))

        # Fix args
        if eval_args.test_batch_size == -1:
            eval_args.test_batch_size = None
        if eval_args.load_step is not None:
            # TODO load from given step
            warnings.warn(
                "Loading weights from specific training step is not supported "
                "for now. The model will be loaded from the last checkpoint.")

        # Get path to load model
        checkpoint_folder = os.path.join('checkpoints', eval_args.load)

        # Add date string and create folder on evaluation_results
        self._result_folder = os.path.join('evaluation_results',
                                           date_str + '_' + eval_args.load)
        self._img_folder = os.path.join(self._result_folder, 'imgs')
        os.makedirs(self._result_folder)
        os.makedirs(self._img_folder)

        # Set img folder for viz module
        viz.img_folder = self._img_folder

        # Load config
        config_path = os.path.join(checkpoint_folder, 'config.pkl')
        with open(config_path, 'rb') as file:
            args = pickle.load(file)

        # Modify config for testing
        if eval_args.test_batch_size is not None:
            args.test_batch_size = eval_args.test_batch_size
        args.dry_run = False

        experiment = experiment_class(args=args)
        experiment.device = device

        experiment.setup(checkpoint_folder)

        self._experiment = experiment
        self._eval_args = eval_args

    def add_required_args(self, parser):
        """Adds arguments required by BaseExperimentManager to the parser.

        Args:
            parser (argparse.ArgumentParser):
        """

        parser.add_argument('--load',
                            type=str,
                            metavar='NAME',
                            default=self._default_run,
                            help="name of the run to be loaded")
        parser.add_argument('--load-step',
                            type=int,
                            dest='load_step',
                            metavar='N',
                            help='step of checkpoint to be loaded (default:'
                            ' last available)')
        parser.add_argument('--test-batch-size',
                            type=int,
                            default=-1,
                            dest='test_batch_size',
                            metavar='N',
                            help='test batch size')
        parser.add_argument('--nocuda',
                            action='store_true',
                            dest='no_cuda',
                            help='do not use cuda')

    def _parse_args(self):
        """Parses command-line arguments defining experiment settings.

        Returns:
            args (argparse.Namespace): Experiment settings
        """
        raise NotImplementedError

    def run(self):
        """Run evaluator."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Run evaluator."""
        self.run()
