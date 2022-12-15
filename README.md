# Improving the Performance of Backward Chained Behavior Trees using Reinforcement Learning

An experimentation environment for Reinforcment Learning in Backward Chained Behavior Trees.

For more details see the related paper on [arxiv:2112.13744](https://arxiv.org/abs/2112.13744).

# How to set up
Run the following command to install the Python libraries required for the examples in this project.

NB! This has only been tested on Python 3.7 and Project Malmo 0.37.0.

### Installation

- All of the pip requirements:
```
pip install --upgrade pip
pip install -r requirements.txt
```
- Malmo 0.37.0:

See [Bootstrapping](#bootstrapping) on how to run Malmo from the pip wheel.

Alternatively, install Malmo locally:
https://github.com/microsoft/malmo/releases

- Torch

If using CPU:

Normal pip has you covered

If using GPU:
```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

- Cuda 11.1 if using GPU:
https://developer.nvidia.com/cuda-11.1.0-download-archive

### Bootstrapping

If the installation for the Malmo wheel was successful, you can run [bootstrap_malmo()](https://github.com/martkartasev/BTBackchainingRL/blob/0ffacf839f9e8bd1c7217bc75bea5c3e0523d79c/malmo_bootstrap.py#L6) to automatically download the necessary files for malmo.

N.B! Sometimes the necessary libraries are not automatically picked up by python. Seems to be an error in how the wheel is set up. In such a case you can add the [library folder](https://github.com/martkartasev/BTBackchainingRL/tree/master/malmolibrary) to your python path.

Once bootstrapping is complete you can run Malmo from [run_malmo()](https://github.com/martkartasev/BTBackchainingRL/blob/0ffacf839f9e8bd1c7217bc75bea5c3e0523d79c/malmo_bootstrap.py#L11).

# How to run
Once you have done a manual installation or bootstrapped, you should be able to run Malmo locally.

Once Malmo is running, you can start one of the test mission configurations in [main](https://github.com/martkartasev/BTBackchainingRL/blob/master/main.py) to train an agent.

Alternatively, you can run one of the [evaluations](https://github.com/martkartasev/BTBackchainingRL/blob/master/evaluations.py) on one of the included results from our own experiments. Archives are available under [GitHub Releases](https://github.com/martkartasev/BTBackchainingRL/releases) for this repository.

# How to view the Stable-Baselines integrated tensorboard

The logging has already been set up. Running the training examples will generate tensorboard files into a separate directory which can be viewed with the following command.

```
tensorboard --logdir <tensorboard directory>
```

