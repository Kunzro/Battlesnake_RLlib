import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import UnifiedLogger, pretty_print
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from datetime import datetime

from battlesnake_gym import BattlesnakeGym
from battlesnake_gym.rewards import SimpleRewards

from policies import *
from ray.rllib.models import ModelCatalog
import os

ModelCatalog.register_custom_model("CustomNetwork", CustomNetwork)
ModelCatalog.register_custom_model("CustomNetworkMax", CustomNetworkMax)
ModelCatalog.register_custom_model("CustomNetworkAverage", CustomNetworkAverage)
ModelCatalog.register_custom_model("CustomNetworkDeeper", CustomNetworkDeeper)
ModelCatalog.register_custom_model("CustomNetworkWider", CustomNetworkWider)
ModelCatalog.register_custom_model("CustomNetworkWiderPool", CustomNetworkWiderPool)
ModelCatalog.register_custom_model("CustomNetworkDeeperPool", CustomNetworkDeeperPool)



def policy_mapping_fn(agent_id, *arg):
    return "single_snake"
map_size = 19
env_config_dict = {
                "observation_type": "compact-flat-51s",
                "map_size": (map_size, map_size),
                "number_of_snakes": 4, 
                "snake_spawn_locations": [],
                "food_spawn_locations": [],
                "verbose": False,
                "initial_game_state": None,
                "rewards": SimpleRewards()
            }
dummy_env = BattlesnakeGym(config=env_config_dict)
obs_space = dummy_env.observation_space
act_space = dummy_env.action_space

ray.init()
def get_config(model):
    config = ppo.DEFAULT_CONFIG.copy()

    # trainer settings
    config["num_gpus"] = 1
    config["num_workers"] = 8
    config["num_envs_per_worker"] = 8
    # carefull this can render a lot of videos
    config["record_env"] = False
    # env configs
    config["env_config"] = env_config_dict
    config["framework"] = "torch"
    # model settings

    config["model"] = {
        "custom_model": model,
        "dim": map_size,
    }

    # multi agent settings
    config["multiagent"] = {
        "policies": {
            "single_snake": (PPOTorchPolicy, obs_space, act_space, {})
        },
        "policy_mapping_fn": policy_mapping_fn
    }
    # set evaluation options
    config["evaluation_interval"] = 5
    config["evaluation_num_episodes"] = 50
    return config

networks = [
    (CustomNetwork, "CustomNetwork"), 
#    (CustomNetworkDeeper, "CustomNetworkDeeper"), 
#    (CustomNetworkWider, "CustomNetworkWider"),
#    (CustomNetworkMax, "CustomNetworkMax"),
#    (CustomNetworkAverage, "CustomNetworkAverage"),
#    (CustomNetworkDeeperPool, "CustomNetworkDeeperPool"),
#    (CustomNetworkWiderPool, "CustomNetworkWiderPool")
    ]


print(torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i))
for network, networkName in networks:

    def logger_creator(config):
        date_str = datetime.today().strftime("%Y-%m-%d")
        logdir_prefix = "{}_{}_{}".format("PPO", networkName, date_str)
        home_dir = os.path.expanduser("~/ray_results")
        logdir = os.path.join(home_dir, logdir_prefix)
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)

    config=get_config(networkName)
    trainer = PPOTrainer(config=config, env=BattlesnakeGym, logger_creator=logger_creator)
    
    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(1001):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint = trainer.save(checkpoint_dir=f"/home/sem21h18/ray_results/PPO_{networkName}_{date_str}")
            print("checkpoint saved at", checkpoint)