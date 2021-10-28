# Behavior Trees Back-chaining with Reinforcement Learning 

# How to set up
Run the following command to install the Python libraries required for the examples in this project.

NB! This has only been tested on Python 3.7 and Project Malmo 0.37.0.

You will need to install 

- Malmo 0.37.0:

https://github.com/microsoft/malmo/releases

- All of the pip requirements:
```
pip install --upgrade pip
pip install -r requirements.txt
```
- Torch

If using CPU:
```
pip3 install torch==1.8.2+cpu torchvision==0.9.1+cpu torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html```
```

If using GPU:
```
 pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

- Cuda 11.1 if using GPU:
https://developer.nvidia.com/cuda-11.1.0-download-archive


# How to run
Run one of the test mission configurations under "Experiments".

# How to use tensorboard

```
tensorboard --logdir <tensorboard directory>
```
