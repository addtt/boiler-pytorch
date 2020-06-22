import os
import pickle
import timeit
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

from boilr.nn.utils import global_norm, grad_global_norm
from boilr.options import get_option
from boilr.utils import set_rnd_seed, get_date_str
from boilr.utils import viz
from boilr.utils.summarize import History, SummarizerCollection


class Trainer:
    """Generic tool for training models.

    All model- and experiment-specific work is defined and performed in an
    experiment object, which is provided at initialization.

    Args:
        experiment (boilr.experiments.BaseExperimentManager)
    """

    def __init__(self, experiment):
        self.experiment = experiment
        resume = experiment.args.resume != ""

        if resume:
            # Folder string = run name = resume argument
            folder_str = experiment.args.resume
            print("Resume from '{}'".format(folder_str))
            msg = "When resuming training, the optimizer's state is not restored"
            warnings.warn(msg)
            msg = ("When resuming training, the relative time on tensorboard "
                   "has a gap due to wall clock timestamps")
            warnings.warn(msg)

            # Get all folder names to resume saving results
            tboard_folder = self._setup_paths(folder_str)
            self.tb_writer = None
            if have_tensorboard:  # maybe we didn't have tboard originally
                os.makedirs(tboard_folder, exist_ok=True)
                self.tb_writer = tensorboard.SummaryWriter(tboard_folder)

            # Forget about all arguments, load all of them from saved config
            # (the 'resume' argument is overwritten in the process)
            config_path = os.path.join(self.checkpoint_folder, 'config.pkl')
            experiment.load_args_from_pickle(config_path)

            # Load training and test history from log.pkl
            with open(self.log_path, 'rb') as file:
                history = pickle.load(file)
                self.train_history = History(history['train'])
                self.test_history = History(history['test'])

        args = experiment.args

        assert args.checkpoint_every % args.test_log_every == 0

        # Pick device (cpu/cuda)
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

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
            folder_str = date_str
            run_descr = experiment.run_description
            if len(run_descr) > 0:
                folder_str += '_' + experiment.run_description
            tboard_folder = self._setup_paths(folder_str)
            self.tb_writer = None
            if not args.dry_run:
                os.makedirs(self.img_folder)
                os.makedirs(self.checkpoint_folder)
                if have_tensorboard:
                    os.makedirs(tboard_folder)
                    self.tb_writer = tensorboard.SummaryWriter(tboard_folder)
                config_path = os.path.join(self.checkpoint_folder, 'config.pkl')
                with open(config_path, 'wb') as fd:
                    pickle.dump(args, fd)

        viz.img_folder = self.img_folder
        experiment.setup(
            device=self.device,
            checkpoint_folder=self.checkpoint_folder if resume else None)

        # Check everything is initialized properly
        self._check_experiment(experiment)

    def _setup_paths(self, folder_str):
        result_folder = os.path.join('output', folder_str, 'results')
        self.img_folder = os.path.join(result_folder, 'imgs')
        self.checkpoint_folder = os.path.join('output', folder_str,
                                              'checkpoints')
        self.log_path = os.path.join(result_folder, 'log.pkl')
        tboard_folder = os.path.join('output', folder_str, 'tensorboard_logs')
        return tboard_folder

    def run(self):
        """Runs the trainer."""

        # This is needed when loading to resume training
        first_step = True

        # Setup
        e = self.experiment
        train_loader = e.dataloaders.train
        progress = None
        show_progress = get_option('show_progress_bar')
        train_summarizers = SummarizerCollection(
            mode='moving_average',
            ma_length=get_option('train_summarizer_ma_length'))
        # Additional summarizers are considered independent of the train/test
        # regime, they are not printed, they are saved to tensorboard only once
        # (during training and not testing), and for now they are not saved to
        # the history.
        additional_summarizers = SummarizerCollection(
            mode='moving_average',
            ma_length=get_option('train_summarizer_ma_length'))

        # Training mode
        e.model.train()

        # Main loop
        for epoch in range(1, e.args.max_epochs + 1):
            for batch_idx, (x, y) in enumerate(train_loader):

                step = e.model.global_step

                if step >= e.args.max_steps:
                    break

                if step % e.args.test_log_every == 0:

                    # Test model (unless we just resumed training)
                    if not first_step or step == 0:
                        with torch.no_grad():
                            self._test(epoch)

                    # Save model checkpoint (unless we just started/resuming training)
                    if not first_step and step % e.args.checkpoint_every == 0:
                        print("* saving model checkpoint at "
                              "step {}".format(step))
                        e.model.checkpoint(self.checkpoint_folder,
                                           e.args.keep_checkpoint_max)

                    # (Re)start progress bar
                    if show_progress:
                        progress = tqdm(total=e.args.test_log_every,
                                        desc='train')

                    # Restart timer to measure training speed
                    timer_start = timeit.default_timer()
                    steps_start = e.model.global_step

                    # This timer stuff won't make sense if 'test every' is not
                    # a multiple of 'train every'. Which is now true, but let's
                    # be safe in case things change
                    assert e.args.test_log_every % e.args.test_log_every == 0

                # Reset gradients
                e.optimizer.zero_grad()

                # Forward pass: get loss and other info
                outputs = e.forward_pass(x, y)

                # Compute gradients (backward pass)
                outputs['loss'].backward()

                e.post_backward_callback()

                if e.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        e.model.parameters(), max_norm=e.args.max_grad_norm)

                # Add batch metrics to summarizers
                metrics_dict = e.get_metrics_dict(outputs)
                train_summarizers.add(metrics_dict)

                # Compute gradient stats and add to summarizers
                # - grad norm of each parameter
                # - grad norm of given group (or default groups)
                # - total grad norm
                # TODO groups
                grad_stats = {
                    'grad_norm_total/grad_norm_total':
                        grad_global_norm(e.model.parameters())
                }
                for n, p in e.model.named_parameters():
                    k = 'grad_norm_per_parameter/' + n
                    grad_stats[k] = grad_global_norm(p)
                additional_summarizers.add(grad_stats)

                # Compute L2 norm of parameters and add it to summarizers
                additional_summarizers.add(
                    {'L2_norm': global_norm(e.model.parameters())})

                # Update progress bar
                if show_progress:
                    progress.update()

                # Close progress bar if test occurs at next loop iteration
                if (step + 1) % e.args.test_log_every == 0:
                    # If show_progress is False, progress is None
                    if progress is not None:
                        progress.close()

                if (step + 1) % e.args.train_log_every == 0:
                    # step+1 because we already did a forward/backward step

                    # Get moving average of metrics and reset summarizers
                    train_summaries = train_summarizers.get_all(reset=True)
                    additional_summaries = additional_summarizers.get_all(
                        reset=True)

                    # Get training speed and add it to summaries
                    elapsed = timeit.default_timer() - timer_start
                    iterations = e.model.global_step - steps_start
                    steps_per_sec = iterations / elapsed
                    ex_per_sec = steps_per_sec * e.args.batch_size
                    additional_summaries['speed/steps_per_sec'] = steps_per_sec
                    additional_summaries['speed/examples_per_sec'] = ex_per_sec
                    timer_start = timeit.default_timer()
                    steps_start = e.model.global_step

                    # Print summaries
                    print(e.train_log_str(train_summaries, step + 1, epoch))

                    # Add train summaries (smoothed) to history and dump it to
                    # file and to tensorboard if available
                    self.train_history.add(train_summaries, step + 1)
                    if not e.args.dry_run:
                        with open(self.log_path, 'wb') as fd:
                            pickle.dump(self._history_dict(), fd)
                        if self.tb_writer is not None:
                            for k, v in train_summaries.items():
                                self.tb_writer.add_scalar(
                                    'train_' + k, v, step + 1)
                            for k, v in additional_summaries.items():
                                self.tb_writer.add_scalar(k, v, step + 1)

                # Optimization step
                e.optimizer.step()

                # Increment model's global step variable
                e.model.increment_global_step()

                first_step = False

            if step >= e.args.max_steps:
                break

        if progress is not None:  # if show_progress is False, progress is None
            progress.close()
        if step < e.args.max_steps:
            print("Reached epochs limit ({} epochs)".format(e.args.max_epochs))
        else:
            print("Reached steps limit ({} steps)".format(e.args.max_steps))

    def _test(self, epoch):
        e = self.experiment
        step = e.model.global_step

        # Evaluation mode
        e.model.eval()

        # Get test results
        summaries = e.test_procedure()

        # Save images
        if step % e.args.test_imgs_every == 0:
            if not e.args.dry_run:
                e.save_images(self.img_folder)

        # Print log string with test results (experiment-specific)
        print(e.test_log_str(summaries, step, epoch))

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
