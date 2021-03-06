# boiler-pytorch

Basic framework for training stuff in PyTorch. It's quite tailored to projects 
I've been working on lately, so it's meant for personal use. Its sole purpose is 
to do away with `boilr`plate code, and having it here makes it easier to 
share it across projects.

## Install

```shell script
pip install boilr
```


## Usage example/template

There's a usage example that can be useful as template. It's a basic VAE
for MNIST quickly hacked together. The example files are:
- `example.py`
- `example_evaluate.py`
- `experiments/mnist_experiment/data.py`
- `experiments/mnist_experiment/experiment_manager.py`
- `models/mnist_vae.py`

Install requirements and run the example:

```shell script
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python example.py
```

For evaluation:

```shell script
CUDA_VISIBLE_DEVICES=0 python example_evaluate.py --ll --ll-samples 100 --load $RUN_NAME
```
using the name of the folder in `output/` generated from running the example.



## Quick reference


### Built-in functionalities

The following functionalities are available out-of-the-box:
- Easy logging of metrics to tensorboard and to a pickle file. Metrics are collected at every training step, smoothed, and logged/saved at a specified frequency. The amount of smoothing is also customizable.
- Summaries of the metrics are automatically printed after each training and testing phase. This can be easily customized. 
- Training speed, gradient norm (global and per-parameter), and L2 norm of the model parameters are all automatically logged.
- It's easy to save images from testing, in a dedicated folder.
- Gradient clipping (by global norm), controllable through a command-line argument.
- Automatic model checkpointing, with command-line argument to control the maximum number of recent checkpoints to be kept.
- Command-line argument to resume training from checkpoint, and everything is taken care of.
- Progress bar for training and testing, using `tqdm`. Can be switched off.
- Data-dependent initialization (command-line argument).
- Reproducibility: set random seed across all devices and Python libraries.
- A suite of utility classes and methods in the packages `boilr.nn` and `boilr.utils` (most of them for internal use).
In particular `boilr.nn.modules` and `boilr.utils.viz` might be more generally useful.
- A long list of command-line arguments to control some of the behaviour above. 
Some arguments are not directly used, but it's convenient to have them already defined: e.g. if a custom `DataLoader` is necessary, the batch size is easily accessible with `args.batch_size`; and when creating the optimizer, the learning rate is `args.lr`. 
- See `boilr.options` for package-wide options. Usually it's not necessary to change them, but they give some more flexibility.

#### Command-line arguments

There are built-in command-line arguments with default values. These defaults can be easily 
overridden programmatically when making the experiment class that subclasses `boilr`'s. 
The built-in arguments are the following:
- `batch-size`: training batch size (default: None)
- `test-batch-size`: test batch size (default: None)
- `lr`: learning rate (default: None)
- `max-grad-norm`: maximum global norm of the gradient. It is clipped if larger. If None, no clipping is performed. (default: None)
- `seed`: random seed (default: 54321)
- `tr-log-every`: log training metrics every this number of training steps (default: 1000)
- `ts-log-every`: log test metrics every this number of training steps. It must be a multiple of `--tr-log-every` (default: 1000)
- `ts-img-every`: save test images every this number of training steps. It must be a multiple of `--ts-log-every` (default: same as `--ts-log-every`)
- `checkpoint-every`: save model checkpoint every this number of training steps (default: 1000)
- `keep-checkpoint-max`: keep at most this number of most recent model checkpoints (default: 3)
- `max-steps`: max number of training steps (default: 1e10)
- `max-epochs`: max number of training epochs (default: 1e7)
- `nocuda`: do not use cuda (default: False)
- `descr`: additional description for experiment name
- `dry-run`: do not save anything to disk (default: False)
- `resume`: load the run with this name and resume training

Additionally, for `VAEExperimentManager`, the following arguments are available:
- `ll-every`: evaluate log likelihood (with the importance-weighted bound) every this number of training steps (default: 50000)
- `ll-samples`: number of importance-weighted samples to evaluate log likelihood (default: 100)


### Getting started

1. subclass a base dataset manager class;
2. subclass a base model class;
3. subclass a base experiment manager class (the model class is used in here);
4. make a short script that creates the experiment object, uses it to create a `boilr.Trainer`, and runs the trainer;
5. optionally, subclass the base evaluator to set up an "offline" evaluation pipeline.

See below for more details.

#### Dataset manager class (1)

The class `boilr.data.BaseDatasetManager` must be subclassed. The subclass *must* implement
the method `_make_datasets` which should return a tuple `(train, test)` with the training
and test sets as PyTorch `Dataset`s.
A basic implementation of `_make_dataloaders` is already provided, but can be overridden to make
custom data loaders.


#### Model class (2)

One of the model classes must be subclassed to inherit core methods in the base implementation `boilr.models.BaseModel`.
These models also automatically subclass `torch.nn.Module` (so it must implement `forward`).
In addition, `boilr.models.BaseGenerativeModel` (subclassing `BaseModel`) defines a method `sample_prior` that must be implemented by subclasses.


#### Experiment manager class (3)

One of the base experiment classes in `boilr.experiments` must be subclassed. The subclass *must* implement:
- `_make_datamanager` to create the dataset manager, which should subclass `boilr.data.BaseDatasetManager`;
- `_make_model` to create the model, which should subclass `boilr.models.BaseModel`;
- `_make_optimizer` to create the optimizer, which should subclass `torch.optim.optimizer.Optimizer`;
- `forward_pass` to perform a simple single-pass model evaluation and returns losses and metrics;
- `test_procedure` to evaluate the model on the test set (usually heavily based on the `forward_pass` method).

Typically should be overridden:
- `_define_args_defaults`, `_add_args`, and `_check_args` (or a subset of these) to manage parsing of command-line arguments;
- `_make_run_description` which returns a string description of the run, used for output folders;
- `save_images` to save output images (e.g. reconstructions and samples in VAEs).

May be overridden for additional control:
- `post_backward_callback` is called by the `Trainer` after the backward pass but before the optimization step;
- `get_metrics_dict` translates a dictionary of results to a dictionary of metrics to be logged (by default this simply copies over the keys);
- `train_log_str` and `test_log_str` return log strings for test and training metrics.

**Note**: The class `VAEExperimentManager` implements default `test_procedure` and `save_images` 
methods for variational inference with VAEs.


#### Example training script (4)

```python
from boilr import Trainer
from my_experiment import MyExperimentClass

if __name__ == "__main__":
    experiment = MyExperimentClass()
    trainer = Trainer(experiment)
    trainer.run()
```


#### Offline evaluator class (5)

If offline evaluation is necessary, `boilr.eval.BaseOfflineEvaluator` can be subclassed by implementing:

- `run` to run the evaluation;
- as above, `_define_args_defaults`, `_add_args`, and `_check_args` (or a subset of these) to manage parsing of command-line arguments.

The method `run` can be executed by simply calling the evaluator object.
See `example_evaluate.py`.



### Notes

- It also works without `tensorboard`, but it won't save tensorboard logs.
