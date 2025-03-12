import gymnasium as gym
import gymnasium_robotics
from pick_place_custom import CustomPickPlaceEnv

gym.envs.registration.register(
    id="CustomPickPlace-v0",
    entry_point="pick_place_custom:CustomPickPlaceEnv",
)

gym.register_envs(gymnasium_robotics)

# env = gym.make("FetchPickAndPlace-v2", render_mode="human")
env = gym.make("CustomPickPlace-v0", render_mode="human")
# env = gym.make("FetchPickAndPlaceDense-v3", render_mode="human")
obs, info = env.reset()

for _ in range(1000000000000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    print(obs, reward, done, truncated, info)
    # if done:
    #     obs, info = env.reset()
        
env.close()
