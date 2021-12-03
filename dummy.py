from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
import ray.rllib.agents.ppo as ppo

from policies import *
from ray.rllib.models import ModelCatalog

from battlesnake_gym import BattlesnakeGym
from battlesnake_gym.rewards import SimpleRewards

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

config = ppo.DEFAULT_CONFIG.copy()
config["model"] = {
    "custom_model": "CustomNetwork"
}
config["env_config"] = env_config_dict
config["framework"] = "torch"
config["multiagent"] = {
    "policies": {
        "single_snake": (PPOTorchPolicy, obs_space, act_space, {})
    },
    "policy_mapping_fn": policy_mapping_fn
}
trainer = PPOTrainer(config=config, env=BattlesnakeGym)
trainer.load_checkpoint("/home/sem21h18/ray_results/PPO_CustomNetwork_2021-12-02/checkpoint_000061/checkpoint-61")

env = BattlesnakeGym(config["env_config"])
for i in range(20):
    obs = env.reset()
    done = False
    while not done:
        actions = trainer.compute_actions(obs, explore=False, policy_id="single_snake")
        obs, rewards, dones, infos = env.step(actions=actions)
        env.render(mode="human")
        done = dones["__all__"]
