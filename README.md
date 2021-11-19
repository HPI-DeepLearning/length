# LENGTH - Lightning-fast Extensible Neural-network Guarding The HPI

![logo](https://github.com/HPI-DeepLearning/length/raw/master/logo.gif)

A simple neural network implementation that could be used for education.
But beware! Some important parts of the code are missing :wink: and you
have to fill the blanks!

# Installation

1. Make sure to install `Python 3` on your device
  - Windows: Get it [here](https://www.python.org/downloads/windows/)
  - Mac: Get it [here](https://www.python.org/downloads/mac-osx/) or use
  your favourite package manager
  - Linux: Use your favourite package manager i.e. `pacman -S python` or
  `apt install python3`
2. Create a virtualenvironment
  - you can do so with `python3 -m venv <path to virtualenv>`
  - If you are using Linux, we recommend that you install
  [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
  and organize all virtualenvironments with this tool, its quite neat.
3. Load the virtualenvironment
4. Clone the repository
4. Install all necessary libraries with `pip install -r requirements.txt`


# Structure of the Repository

You can find all neural network specific code in the directory `length`.
Follows is a list of all directories and their respective content:
- `data_sets` contains code that is necessary for dealing with specific
datasets like `MNIST` or `FashionMNIST`. Those implementations shall
follow the interface specification in `length/data_set.py`.
- `functions` contains code for all neural network functions that do not
have any parameters that need to be updated. All those functions follow
the interface specified in `length/function.py`.
- `initializers` contains code for weight initializers that follow the
initializer interface specified in `length/initializer.py`.
- `layers` contains code for all implemented neural network layers.
Those layers follow the interface specification in `length/layer.py`. A
layer is a neural network function that has internal parameters, which
need to be updated.
- `models` contains code for sample models that come with this library
and can be trained out of the box.
- `optimizers` contains the implementation of all neural network
optimizers. Those implementations follow the interface, specified in
`length/optimizer.py`.
- `tests` contains tests for the library, that use `py.test` as testing
framework.

The file `train.py` in the root directory contains sample code for
training a neural network with this library.

# Training

You can start a train run, using the sample train script, by entering
the following command from the root directory of the repository:
`python train.py`. If you want to know more about possible command-line
arguments, type `python train.py -h`.


# Principles of this library

This library follows the design principles of
[Chainer](https://chainer.org/) in a very rudimentary way. This library
implements a dynamic computational graph that can be used to easily
prototype, design, and debug neural networks. It furthermore allows for
the design of dynamic neural networks that change at every iteration,
making such an approach suitable for recurrent neural networks.

A word of *WARNING*: If you ever feel tempted to use this library for
training a real neural network and not just a toy example: **don't**! Use a
library like Chainer that is more mature and also supports GPUs. The
transition from this library to Chainer should not be that difficult, as
the same principles are used here and in Chainer.

# Contributions

Contributions in every form are welcome. If you think, that you nicely
implemented a new layer, we are happy to see your contribution in form
of a pull request!

# One Last Thing

**Remember:** Do not use this library for training a real network.
This library is only intended to be used for education and not for real
prototyping.
