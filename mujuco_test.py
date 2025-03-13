import gymnasium as gym
from pick_place_custom import PickPlaceCustomEnv


xml_path = "franka_emika_panda/scene_pick_and_place.xml"
env = PickPlaceCustomEnv(xml_path, render_mode="human")
obs, info = env.reset()

for _ in range(100000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    print(obs, reward, done, truncated, info)
    if done:
        obs, info = env.reset()
        
env.close()
