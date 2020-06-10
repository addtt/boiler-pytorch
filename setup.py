import re
from os.path import join

import setuptools

# Get version string by parsing _version.py
version_fname = join('boilr', '_version.py')
content = open(version_fname, "rt").read()
regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
match = re.search(regex, content, re.M)
if match:
    version_string = match.group(1)
else:
    raise RuntimeError(
        "Unable to find version string in {}".format(version_fname))

# Get long description from README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Define list of packages
packages = setuptools.find_packages()
packages = [p for p in packages if p.startswith('boilr')]

setuptools.setup(name="boilr",
                 version=version_string,
                 author="Andrea Dittadi",
                 author_email="andrea.dittadi@gmail.com",
                 description="Basic framework for training models with PyTorch",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/addtt/boiler-pytorch",
                 packages=packages,
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3.6',
                 install_requires=[
                     'numpy>=1.18',
                     'torch>=1.4',
                     'torchvision>=0.5',
                     'matplotlib>=3.2',
                     'tqdm',
                     'pillow>=7.1',
                 ])
