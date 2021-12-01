from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import gym
from ray.rllib.utils.typing import ModelConfigDict
from typing import Dict, Any, List
import torch
from utils import conv_output_shape
from math import prod
class CustomNetwork(TorchModelV2, nn.Module):
    """Custom CNN Network"""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_channels = obs_space.shape[-1]
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 2, pad = 2)
        post_cnn_size = conv_output_shape(post_cnn_size, (4,4), 2, pad = 1)
        
        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None
        
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].detach().clone().float()
        obs = torch.permute(obs, (0, 3, 1, 2))
        #for i in range(1000):
        #    print(obs.shape)
        x = self.sequential_layers(obs)
        #model_out, self._value_out = self.base_model(
        #    input_dict["obs"])
        self._value = self.value_layer(x)
        return self.output_layer(x), state

    def value_function(self):
        assert self._value is not None, "must call forward() first"
        #for i in range(50):
        #    print(self._value.shape)
        return self._value.squeeze(1)

class CustomNetworkMax(CustomNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = obs_space.shape[-1]
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 1, pad = 2)
        post_cnn_size = conv_output_shape((post_cnn_size[0]//2, post_cnn_size[1]//2), (4,4), 1, pad = 1)
        post_cnn_size = (post_cnn_size[0]//2, post_cnn_size[1]//2)
        
        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=1, padding = 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=1, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None

class CustomNetworkAverage(CustomNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = obs_space.shape[-1]
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 1, pad = 2)
        post_cnn_size = conv_output_shape((post_cnn_size[0]//2, post_cnn_size[1]//2), (4,4), 1, pad = 1)
        post_cnn_size = (post_cnn_size[0]//2, post_cnn_size[1]//2)
        
        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=1, padding = 2),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=1, padding = 1),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None

class CustomNetworkDeeper(CustomNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = obs_space.shape[-1]
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 2, pad = 2)
        post_cnn_size = conv_output_shape(post_cnn_size, (4,4), 2, pad = 1)
        post_cnn_size = conv_output_shape(post_cnn_size, (3,3), 1, pad = 1)
    
        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding = 1),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None

class CustomNetworkWider(CustomNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = obs_space.shape[-1]
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 2, pad = 2)
        post_cnn_size = conv_output_shape(post_cnn_size, (4,4), 2, pad = 1)
    
        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=2, padding = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None

class CustomNetworkWiderPool(CustomNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = obs_space.shape[-1]
     
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 1, pad = 2)
        post_cnn_size = conv_output_shape((post_cnn_size[0]//2, post_cnn_size[1]//2), (4,4), 1, pad = 1)
        post_cnn_size = (post_cnn_size[0]//2, post_cnn_size[1]//2)

        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=1, padding = 2),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=1, padding = 1),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None

class CustomNetworkDeeperPool(CustomNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = obs_space.shape[-1]
     
        post_cnn_size = conv_output_shape((obs_space.shape[0], obs_space.shape[1]), (5,5), 1, pad = 2)
        post_cnn_size = conv_output_shape((post_cnn_size[0]//2, post_cnn_size[1]//2), (4,4), 1, pad = 1)
        post_cnn_size = conv_output_shape((post_cnn_size[0]//2, post_cnn_size[1]//2), (3,3), 1, pad = 1)
        post_cnn_size = (post_cnn_size[0]//2, post_cnn_size[1]//2)

        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=1, padding = 2),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=1, padding = 1),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding = 1),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(prod(post_cnn_size) * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(512,num_outputs)
        self.value_layer = nn.Linear(512,1)
        self._value = None