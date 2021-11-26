import ray
import ray.rllib.agents.dqn as dqn
import os
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy

from battlesnake_gym import BattlesnakeGym
from battlesnake_gym.rewards import SimpleRewards

def policy_mapping_fn(agent_id, *arg):
    return "single_snake"

dummy_env = BattlesnakeGym()
obs_space = dummy_env.observation_space
act_space = dummy_env.action_space

ray.init()
config = dqn.DEFAULT_CONFIG.copy()
# trainer settings
config["num_gpus"] = 1
config["num_workers"] = 1
# config["num_envs_per_worker"] = 1
# carefull this can render a lot of videos
config["record_env"] = False
config["render_env"] = True
# env configs
config["env_config"] = {
                "observation_type": "flat-51s",
                "map_size": (15, 15),
                "number_of_snakes": 4, 
                "snake_spawn_locations": [],
                "food_spawn_locations": [],
                "verbose": False,
                "initial_game_state": None,
                "rewards": SimpleRewards()
            }
config["framework"] = "torch"
# model settings
config["model"] = {
    "dim": 15,
    "no_final_linear": False,
    "conv_activation": "relu",
    "conv_filters": [[32, 5, 1], [32, 3, 2], [32, 3, 2], [32, 4, 1]]
}
# multi agent settings
config["multiagent"] = {
    "policies": {
        "single_snake": (DQNTorchPolicy, obs_space, act_space, {})
    },
    "policy_mapping_fn": policy_mapping_fn
}
# set evaluation options
config["evaluation_interval"] = 5

trainer = DQNTrainer(config=config, env=BattlesnakeGym)

# Can optionally call trainer.restore(path) to load a checkpoint.
usr = os.path.expanduser('~')
ckpt_dir = os.path.join(usr, "ray_results", "DQN_BattlesnakeGym_first_try", "checkpoint_000401", "checkpoint-401")
trainer.load_checkpoint(checkpoint_path=ckpt_dir)

env = BattlesnakeGym(config["env_config"])
for i in range(20):
    obs = env.reset()
    done = False
    while not done:
        actions = trainer.compute_actions(obs, explore=False, policy_id="single_snake")
        obs, rewards, dones, infos = env.step(actions=actions)
        env.render(mode="human")
        done = dones["__all__"]
