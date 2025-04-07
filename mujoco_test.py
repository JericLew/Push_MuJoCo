import gymnasium as gym
from push_custom import PickPlaceCustomEnv
from push_nn import PushNN
import time

xml_path = "franka_emika_panda/scene_push.xml"
env = PickPlaceCustomEnv(xml_path, render_mode="camera")
obs, info = env.reset()
push_nn = PushNN()

for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)

    push_nn.forward(obs)

    env.render()
    print(f"Step: {step}")
    if reward > 1e-3 or reward < -1e-3:
        print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")
        print(f"EE Pose: {obs['state'][7:]}")
        print(f"Reward: {reward} | Done: {done} | Truncated: {truncated}")
    if done:
        obs, info = env.reset()

env.close()
