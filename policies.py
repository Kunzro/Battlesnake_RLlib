from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import gym
from ray.rllib.utils.typing import ModelConfigDict
from typing import Dict, Any, List
import torch
class CustomNetwork(TorchModelV2, nn.Module):
    """Custom CNN Network"""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        nn.Module.__init__(self)

        in_channels = obs_space.shape[-1]
        #for i in range(1000):
        #    print(obs_space.shape)
        self.sequential_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=1, padding = 'same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=1, padding = 'same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7200, 1024),
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