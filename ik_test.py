import numpy as np
import inverse_kinematics as ik

import gymnasium as gym
from push_nn import PushNN


gym.register(
    id="PickPlaceCustomEnv-v0",
    entry_point="push_custom:PickPlaceCustomEnv",
    max_episode_steps=1000,
)

xml_path = "franka_emika_panda/scene_push.xml"
env = gym.make("PickPlaceCustomEnv-v0",
               xml_path=xml_path,
               render_mode="camera",
               max_episode_steps=1000)
obs, info = env.reset()
print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")

for step in range(1000):
    print(f"Step: {step}")
    mjmodel = env.unwrapped.model
    mjdata = env.unwrapped.data
    site_name = 'end_effector'
    target_pos = np.array([0.8, 0.25, 0.2])
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    ikresults = ik.qpos_from_site_pose(mjmodel=mjmodel, mjdata=mjdata, site_name=site_name, target_pos=target_pos, joint_names=joint_names)
    print(f"IK results: {ikresults}")
    action = ikresults[0][:7]

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"EE pos {obs['state'][7:]} | Target pos {target_pos}")

    env.render()
    # if reward > 1e-3 or reward < -1e-3:
    print(f"EE Pose: {obs['state'][7:]}")
    print(f"Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
    # if terminated or truncated:
    #     obs, info = env.reset()
    print(f"=="*20)

env.close()
