import numpy as np
from battlesnake_gym import BattlesnakeGym
from battlesnake_gym.rewards import SimpleRewards

config = {
    "observation_type": "compact-bordered-51s",
    "map_size": (11, 11),
    "number_of_snakes": 4, 
    "snake_spawn_locations": [],
    "food_spawn_locations": [],
    "verbose": False,
    "initial_game_state": None,
    "rewards": SimpleRewards()
}

env = BattlesnakeGym(config=config)
obs = env.reset()
#env.render()
moves = np.array([0, 0, 0, 0])
for i in range(100):
    obs, rewards, dones, infos = env.step(moves)
    #env.render()
print("test")