# Behavior Trees Back-chaining with Reinforcement Learning 

# How to setup
Run the following command to install the Python libraries required for the examples in this project.

You will need to install Cuda 11.1:
https://developer.nvidia.com/cuda-11.1.0-download-archive

```
pip install -r requirements.txt
```
NB! This has been tested on Python 3.7 and Project Malmo 0.37.0. Please install malmo using instructions from https://github.com/microsoft/malmo

If you dont have a GPU available or want to use your CPU instead, modify the torch requirements from requirements.txt to '+cpu' or install torch with the following command instead.
```
pip3 install torch==1.8.2+cpu torchvision==0.9.1+cpu torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html```
```

# How to run
Run one of the test mission configurations under "Experiments".

# How to use tensorboard

```
tensorboard --logdir <tensorboard directory>
```
