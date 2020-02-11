import os
import pickle
import warnings
from os import path

import torch
import torch.utils.data
import torch.utils.data

from .utils import get_date_str


class BaseOfflineEvaluator:  # TODO test this and/or use it in example.py
    """
    Class with boilerplate code to run evaluation routines on a trained model.

    Initialize with the experiment class used for training (the class, not the
    instance), and optionally with the string description of a run that is
    going to be the default run to load from.

    A subclass must define:
    - the _parse_args() method, which is called during init, and must include a
      call to add_required_args() with the argparser as argument.
    - the run() method, defining the whole evaluation procedure.

    The evaluator can then be called directly because it implements __call__().

    The following data attributes are defined, and can be used by subclasses:
    - self.experiment
    - self.img_folder
    - self.eval_args
    """

    def __init__(self, experiment_class, default_run=""):
        eval_args = self._parse_args()
        self.default_run = default_run

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
                "Loading weights from specific training step is not supported for "
                "now. The model will be loaded from the last checkpoint.")

        # Get path to load model
        checkpoint_folder = path.join('checkpoints', eval_args.load)

        # Add date string and create folder on evaluation_results
        result_folder = path.join('evaluation_results',
                                  date_str + '_' + eval_args.load)
        self.img_folder = os.path.join(result_folder, 'imgs')
        os.makedirs(result_folder)
        os.makedirs(self.img_folder)

        # Load config
        config_path = path.join(checkpoint_folder, 'config.pkl')
        with open(config_path, 'rb') as file:
            args = pickle.load(file)

        # Modify config for testing
        if eval_args.test_batch_size is not None:
            args.test_batch_size = eval_args.test_batch_size
        args.dry_run = False

        experiment = experiment_class(args=args)
        experiment.device = device

        experiment.setup(checkpoint_folder)

        self.experiment = experiment
        self.eval_args = eval_args


    def add_required_args(self, parser):
        parser.add_argument('--load',
                            type=str,
                            metavar='NAME',
                            default=self.default_run,
                            help="name of the run to be loaded")
        parser.add_argument('--load-step',
                            type=int,
                            dest='load_step',
                            metavar='N',
                            help='step of checkpoint to be loaded (default: last'
                                 'available)')
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
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.run()
