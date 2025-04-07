import gymnasium as gym
from push_custom import PickPlaceCustomEnv
# from push_nn import PushNN
import time
import mujoco
# from dm_control.mujoco.wrapper import mjbindings
import inverse_kinematics as ik
import numpy as np


xml_path = "franka_emika_panda/scene_push.xml"
env = PickPlaceCustomEnv(xml_path, render_mode="camera")
obs, info = env.reset()

for _ in range(10000000):
    # action = env.action_space.sample()  # Random action
    mjmodel = env.model
    mjdata = env.data
    site_name = 'end_effector'
    target_pos = np.array([0.8, 0.25, 0.2])
    # joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    ikresults = ik.qpos_from_site_pose(mjmodel=mjmodel, mjdata=mjdata, site_name=site_name, target_pos=target_pos)
    print(f"IK results: {ikresults}")
    action = ikresults[0][:7]
    obs, reward, done, truncated, info = env.step(action)
    print(f"EE pos {obs['state'][7:]} | Target pos {target_pos}")

    # push_nn = PushNN()
    # push_nn.forward(obs)

    env.render()
    if reward > 1e-3 or reward < -1e-3:
        print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")
        print(f"EE Pose: {obs['state'][7:]}")
        print(f"Reward: {reward} | Done: {done} | Truncated: {truncated}")
    if done:
        obs, info = env.reset()


env.close()
