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
               render_mode="human",
               max_episode_steps=1000)
obs, info = env.reset()
print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")

for step in range(100000):
    print(f"Step: {step}")
    mjmodel = env.unwrapped.model
    mjdata = env.unwrapped.data
    site_name = 'end_effector'
    target_pos = np.array([0.34, 0, 0.25]) # TARGET START
    target_quat = np.array([0, 0.7071068, -0.7071068, 0]) # TARGET STRAGIHT UP
    target_pos = np.array([0.75, 0.20, 0.25]) # TO RED CORNER
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    ikresults = ik.qpos_from_site_pose(mjmodel=mjmodel, mjdata=mjdata, site_name=site_name, target_pos=target_pos, target_quat=target_quat, joint_names=joint_names)
    print(f"IK results: {ikresults}")
    action = ikresults[0][:7]

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    print(f"Joint Angles: {obs['state'][:7]}")
    print(f"EE Pose: {obs['state'][7:]} | Target Pose: {target_pos}")
    print(f"Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
    # if terminated or truncated:
    #     obs, info = env.reset()
    print(f"=="*20)

env.close()
