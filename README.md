# boiler-pytorch

Basic framework for training stuff in PyTorch. It's quite tailored to projects 
I've been working on lately, so it's meant for personal use. Its sole purpose is 
to do away with *boiler*plate code, and having it here makes it easier to 
share it across projects.

## Install

```shell script
pip install boilr
```

## Usage example/template

There's a usage example that can be useful as template. It's a basic VAE
for MNIST quickly hacked together. The example files/folders are:
- `example.py`
- `models/`
- `experiments/`

Install requirements and run the example:

```shell script
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python example.py
```

Requirements:

- `python 3.7`
- `numpy 1.17.4`
- `matplotlib 3.1.2`
- `torch 1.4.0`
- `torchvision 0.5.0`
- `tensorboard 2.1.0` (it also works without, but it won't save tensorboard logs)
- `tqdm 4.40.2`
