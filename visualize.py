import ray
import numpy as np
from ray.rllib.agents import ppo, dqn, sac
import os
import time
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models import ModelCatalog

from battlesnake_gym import BattlesnakeGym
from battlesnake_gym.rewards import SimpleRewards

from policies import *

ModelCatalog.register_custom_model("CustomNetwork", CustomNetwork)
ModelCatalog.register_custom_model("CustomNetworkMax", CustomNetworkMax)
ModelCatalog.register_custom_model("CustomNetworkAverage", CustomNetworkAverage)
ModelCatalog.register_custom_model("CustomNetworkDeeper", CustomNetworkDeeper)
ModelCatalog.register_custom_model("CustomNetworkWider", CustomNetworkWider)
ModelCatalog.register_custom_model("CustomNetworkWiderPool", CustomNetworkWiderPool)
ModelCatalog.register_custom_model("CustomNetworkDeeperPool", CustomNetworkDeeperPool)
#ModelCatalog.register_custom_model("CustomNetworkLarge", CustomNetworkLarge)

def policy_mapping_fn(agent_id, *arg):
    return "single_snake"

ray.init()

map_size = 19
observation_type = "compact-flat-51s" # compact-bordered-51s or compact-flat-51s
env_config_dict = {
                "observation_type": observation_type,
                "map_size": (map_size, map_size),
                "number_of_snakes": 3, 
                "snake_spawn_locations": [],
                "food_spawn_locations": [],
                "verbose": False,
                "initial_game_state": None,
                "rewards": SimpleRewards()
            }

def trainer_config(algo, policy):
    config = algo.DEFAULT_CONFIG.copy()
    # trainer settings
    config["num_gpus"] = 1
    config["num_workers"] = 1
    # config["num_envs_per_worker"] = 1
    # carefull this can render a lot of videos
    # env configs
    config["env_config"] = env_config_dict
    dummy_env = BattlesnakeGym(config=env_config_dict)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    config["framework"] = "torch"
    # model settings
    if algo == sac:
        config["Q_model"] = {
            "custom_model": "CustomNetwork",
            "custom_model_config": {
                "observation_type": observation_type
            }
        }
        config["policy_model"] = {
            "custom_model": "CustomNetwork",
            "custom_model_config": {
                "observation_type": observation_type
            }
        }
    else:
        config["model"] = {
            "custom_model": "CustomNetwork",
            "custom_model_config": {
                "observation_type": observation_type
            }
        }
    # multi agent settings
    config["multiagent"] = {
        "policies": {
            "single_snake": (policy, obs_space, act_space, {})
        },
        "policy_mapping_fn": policy_mapping_fn
    }
    config["explore"] = False
    return config

trainer_sac = SACTrainer(config=trainer_config(sac, SACTorchPolicy), env=BattlesnakeGym)
trainer_ppo = PPOTrainer(config=trainer_config(ppo, PPOTorchPolicy), env=BattlesnakeGym)
trainer_dqn = DQNTrainer(config=trainer_config(dqn, DQNTorchPolicy), env=BattlesnakeGym)

# Can optionally call trainer.restore(path) to load a checkpoint.
usr = os.path.expanduser('~')
ckpt_dir_sac = os.path.join(usr, "ray_results", "SAC_CustomNetwork_2021-12-05", "checkpoint-1001")
ckpt_dir_ppo = os.path.join(usr, "ray_results", "PPO_CustomNetwork_2021-12-05", "checkpoint-1001")
ckpt_dir_dqn = os.path.join(usr, "ray_results", "DQN_CustomNetwork_2021-12-05", "checkpoint-1001")

trainer_sac.load_checkpoint(checkpoint_path=ckpt_dir_sac)
trainer_ppo.load_checkpoint(checkpoint_path=ckpt_dir_ppo)
trainer_dqn.load_checkpoint(checkpoint_path=ckpt_dir_dqn)

env = BattlesnakeGym(env_config_dict)
for i in range(20):
    obs = env.reset()
    done = False
    while not done:
        actions_sac = trainer_sac.compute_actions(obs, explore=False, policy_id="single_snake")
        actions_ppo = trainer_ppo.compute_actions(obs, explore=False, policy_id="single_snake")
        actions_dqn = trainer_dqn.compute_actions(obs, explore=False, policy_id="single_snake")
        actions = np.zeros((3), dtype=np.int64)
        if 0 in actions_sac:
            actions[0] = actions_sac[0]
        if 1 in actions_ppo:
            actions[1] = actions_ppo[1]
        if 2 in actions_dqn:
            actions[2] = actions_dqn[2]
        obs, rewards, dones, infos = env.step(actions=actions)
        env.render(mode="human")
        time.sleep(0)
        done = dones["__all__"]
        if done:
            time.sleep(2)
        if 0 in dones and dones[0]:
            print(f"SAC infos: {infos[0]}")
        if 1 in dones and dones[1]:
            print(f"PPO infos: {infos[1]}")
        if 2 in dones and dones[2]:
            print(f"DQN infos: {infos[2]}")
    winner = "no one"
    if 0 in dones and not dones[0]:
        winner = "SAC"
    if 1 in dones and not dones[1]:
        winner = "PPO"
    if 2 in dones and not dones[2]:
        winner = "DQN"
    print(f"Game ended, winner: {winner}")
    print()
