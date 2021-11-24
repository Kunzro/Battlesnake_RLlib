import numpy as np
from battlesnake_gym import BattlesnakeGym

env = BattlesnakeGym()
obs = env.reset()
env.render()
moves = np.array([0, 0, 0, 0])
for i in range(100):
    obs, rewards, dones, infos = env.step(moves)
    env.render()
print("test")