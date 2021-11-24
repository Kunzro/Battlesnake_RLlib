from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import gym
from ray.rllib.utils.typing import ModelConfigDict

class CustomNetwork(TorchModelV2, nn.Module):
    """Custom CNN Network"""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_channels = obs_space.shape[0]

        self.sequentlial_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=1)
        )