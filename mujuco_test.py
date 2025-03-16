import gymnasium as gym
from push_custom import PickPlaceCustomEnv
import time

xml_path = "franka_emika_panda/scene_push.xml"
env = PickPlaceCustomEnv(xml_path, render_mode="camera")
obs, info = env.reset()

for _ in range(10000000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if reward > 1e-3 or reward < -1e-3:
        print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")
        print(f"EE Pose: {obs['state'][7:]}")
        print(f"Reward: {reward} | Done: {done} | Truncated: {truncated}")
    if done:
        obs, info = env.reset()


env.close()
