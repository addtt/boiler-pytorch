import os
import pickle
import warnings

import torch
try:
    from torch.utils import tensorboard
    have_tensorboard = True
except ImportError as e:
    msg = "{}: {}".format(e.__class__.__name__, e)
    msg = "Could not import tensorboard.\n" + msg
    warnings.warn(msg)
    have_tensorboard = False
from tqdm import tqdm

from .summarize import History, SummarizerCollection
from .utils import set_rnd_seed, get_date_str


class Trainer:
    """
    Generic tool for training models. All model- and experiment-specific work is
    defined and performed in an experiment object, which is provided at init.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        args = experiment.args
        resume = args.resume != ""

        if resume:
            # Folder string = run name = resume argument
            folder_str = args.resume
            print("Resume from '{}'".format(folder_str))
            print("[!] When resuming training, the optimizer's state is not restored")
            warnings.warn("Tensorboard logs are stored in the same folder as "
                          "the original run we're resuming. Not clear if this "
                          "overwrites the previous records.")

            # Get all folder names to resume saving results
            result_folder = os.path.join('results', folder_str)
            self.img_folder = os.path.join(result_folder, 'imgs')
            self.checkpoint_folder = os.path.join('checkpoints', folder_str)
            self.log_path = os.path.join(result_folder, 'log.pkl')
            tboard_folder = os.path.join('tensorboard_logs', folder_str)
            self.tb_writer = None
            if have_tensorboard:  # maybe we didn't have tboard originally
                os.makedirs(tboard_folder, exist_ok=True)
                self.tb_writer = tensorboard.SummaryWriter(tboard_folder)

            # Forget about all arguments, load all of them from saved config
            # (the 'resume' argument is overwritten in the process)
            config_path = os.path.join(self.checkpoint_folder, 'config.pkl')
            with open(config_path, 'rb') as file:
                args = pickle.load(file)
            assert not args.dry_run  # this would not make sense
            experiment.args = args  # copy them over to the experiment

            # Load training and test history from log.pkl
            with open(self.log_path, 'rb') as file:
                history = pickle.load(file)
                self.train_history = History(history['train'])
                self.test_history = History(history['test'])

        assert args.checkpoint_interval % args.test_log_interval == 0


        # Pick device (cpu/cuda)
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        experiment.device = self.device  # copy device to experiment manager

        if not resume:

            # To save training and test metrics
            self.train_history = History()
            self.test_history = History()

            # Random seed
            set_rnd_seed(args.seed)

            # Get starting time and date for logging
            date_str = get_date_str()

            # Print info
            print('Device: {}, start time: {}'.format(self.device, date_str))

            # Create folders for logs and results, save config
            folder_str = date_str + '_' + experiment.run_description
            result_folder = os.path.join('results', folder_str)
            self.img_folder = os.path.join(result_folder, 'imgs')
            self.checkpoint_folder = os.path.join('checkpoints', folder_str)
            self.log_path = os.path.join(result_folder, 'log.pkl')
            tboard_folder = os.path.join('tensorboard_logs', folder_str)
            self.tb_writer = None
            if not args.dry_run:
                os.makedirs(result_folder)
                os.makedirs(self.img_folder)
                os.makedirs(self.checkpoint_folder)
                if have_tensorboard:
                    os.makedirs(tboard_folder)
                    self.tb_writer = tensorboard.SummaryWriter(tboard_folder)
                config_path = os.path.join(self.checkpoint_folder, 'config.pkl')
                with open(config_path, 'wb') as fd:
                    pickle.dump(args, fd)

        experiment.setup(self.checkpoint_folder if resume else None)

        # Check everything is initialized properly
        self._check_experiment(experiment)


    def run(self):

        # This is needed when loading to resume training
        first_step = True

        # Setup
        e = self.experiment
        train_loader = e.dataloaders.train
        train_summarizers = SummarizerCollection(
            mode='moving_average', ma_length=1000)
        progress = None

        # Training mode
        e.model.train()

        # Main loop
        for epoch in range(1, e.max_epochs + 1):
            for batch_idx, (x, y) in enumerate(train_loader):

                step = e.model.global_step
                if step % e.args.test_log_interval == 0:

                    # Test model (unless we just resumed training)
                    if not first_step or step == 0:
                        with torch.no_grad():
                            self._test(epoch)

                    # Save model checkpoint (unless we just started/resuming training)
                    if not first_step and step % e.args.checkpoint_interval == 0:
                        print("* saving model checkpoint at step {}".format(step))
                        e.model.checkpoint(self.checkpoint_folder)

                    # Restart progress bar
                    progress = tqdm(total=e.args.test_log_interval, desc='train')

                # Reset gradients
                e.optimizer.zero_grad()

                # Forward pass: get loss and other info
                outputs = e.forward_pass(x, y)

                # Compute gradients (backward pass)
                outputs['loss'].backward()

                e.post_backward_callback()

                # Add batch metrics to summarizers
                metrics_dict = e.get_metrics_dict(outputs)
                train_summarizers.add(metrics_dict)

                # Update progress bar
                progress.update()

                # Close progress bar if test occurs at next loop iteration
                if (step + 1) % e.args.test_log_interval == 0:
                    if progress is not None:
                        progress.close()

                if (step + 1) % e.args.log_interval == 0:
                    # step+1 because we already did a forward/backward step

                    # Get moving average of training metrics and reset summarizers
                    summaries = train_summarizers.get_all(reset=True)

                    # Print summaries
                    e.print_train_log(step + 1, epoch, summaries)

                    # Add train summaries (smoothed) to history and dump it to
                    # file and to tensorboard if available
                    self.train_history.add(summaries, step)
                    if not e.args.dry_run:
                        with open(self.log_path, 'wb') as fd:
                            pickle.dump(self._history_dict(), fd)
                        if self.tb_writer is not None:
                            for k, v in summaries.items():
                                self.tb_writer.add_scalar('train_' + k, v, step)

                # Optimization step
                e.optimizer.step()

                # Increment model's global step variable
                e.model.increment_global_step()

                first_step = False

    def _test(self, epoch):
        e = self.experiment
        step = e.model.global_step

        # Evaluation mode
        e.model.eval()

        # Get test results
        summaries = e.test_procedure()

        # Additional testing, possibly including saving images
        e.additional_testing(self.img_folder)

        # Print log string with test results (experiment-specific)
        e.print_test_log(summaries, step, epoch)

        # Save summaries to history
        self.test_history.add(summaries, step)

        if not e.args.dry_run:

            # Save summaries to tensorboard
            if self.tb_writer is not None:
                for k, v in summaries.items():
                    self.tb_writer.add_scalar('validation_' + k, v, step)

            # Save history to file
            with open(self.log_path, 'wb') as fd:
                pickle.dump(self._history_dict(), fd)

        # Training mode
        e.model.train()


    def _history_dict(self):
        return {
            'train': self.train_history.get_dict(),
            'test': self.test_history.get_dict(),
        }


    @staticmethod
    def _check_experiment(e):
        attributes = [
            e.device,
            e.dataloaders,
            e.model,
            e.args,
            e.run_description,
            e.optimizer,
        ]
        assert not (None in attributes)
