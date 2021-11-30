import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
#from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from battlesnake_gym import BattlesnakeGym
from battlesnake_gym.rewards import SimpleRewards

from CustomVisionNet import VisionNetworkWithPooling
from policies import *
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("VisionNetworkWithPooling", VisionNetworkWithPooling)
ModelCatalog.register_custom_model("CustomNetwork", CustomNetwork)
ModelCatalog.register_custom_model("CustomNetworkMax", CustomNetworkMax)
ModelCatalog.register_custom_model("CustomNetworkDeeper", CustomNetworkDeeper)
ModelCatalog.register_custom_model("CustomNetworkWider", CustomNetworkWider)
ModelCatalog.register_custom_model("CustomNetworkWiderPool", CustomNetworkWiderPool)
ModelCatalog.register_custom_model("CustomNetworkDeeperPool", CustomNetworkDeeperPool)



def policy_mapping_fn(agent_id, *arg):
    return "single_snake"
map_size = 19
env_config_dict = {
                "observation_type": "flat-51s",
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
        #"custom_model" : "VisionNetworkWithPooling",
        "custom_model" : model,
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

networks = [(CustomNetwork, "CustomNetwork"), 
            (CustomNetworkDeeper, "CustomNetworkDeeper"), 
            (CustomNetworkWider, "CustomNetworkWider"),
            (CustomNetworkMax, "CustomNetworkMax"),
            (CustomNetworkAverage, "CustomNetworkAverage"),
            (CustomNetworkDeeperPool, "CustomNetworkDeeperPool"),
            (CustomNetworkWiderPool, "CustomNetworkWiderPool")]

for network, networkName in networks:
    trainer = PPOTrainer(config=get_config("CustomNetworkDeeper"), env=BattlesnakeGym)
    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(2):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Trainer's Policy's ModelV2
# (tf or torch) by doing:
# trainer.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch trainers/models.