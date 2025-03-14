import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

# 2ms timesetp
# model.opt.timestep

'''
MuJoCo Notes
1. load model from xml (this describes the robot and the environment)
2. create data from model (this stores the state of the simulation)
3. while data.time < total_sim_time:
    option 1: set data.ctrl to desired control inputs then call mj_step(model, data)
    option 2: add callbacks to model and call mj_step(model, data)
    option 3: do mj_step1(model, data) and mj_step2(model, data) manually

TODO
Add image obs
change action space to x,y,z, open/close gripper
Add random starts
Change to render instead of viewer
'''

class PickPlaceCustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, xml_path, render_mode="human"):
        super().__init__()

        self.colors = ['red', 'green', 'blue']
        self.color_map = {
            'red': [1.0, 0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0]
        }

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Define action and observation space
        action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
        obs_dim = action_dim + 3 # number of joints + EE pos

        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=low, high=high, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        print("Action space:", self.action_space)
        print("Observation space:", self.observation_space)

        # For rewards
        self.initial_object_pos = None
        self.initial_object_quat = None
        self.target_pos = None
        self.prev_object_pos = None
        self.prev_object_quat = None

        # Rendering
        self.render_mode = render_mode
        self.viewer = None

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe('home').qpos
        self.data.ctrl[:] = self.model.keyframe('home').ctrl

        # Randomly select object color
        object_color_name = np.random.choice(self.colors)
        object_color = self.color_map[object_color_name]
        self.model.geom("object_geom").rgba = object_color

        # Randomly offset object position
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[object_body_id]]

        object_xy_delta_pos = np.random.uniform(-0.2, 0.2, size=(2,))
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += object_xy_delta_pos
        self.data.qpos[object_qpos_addr : object_qpos_addr + 3] = new_object_pos  

        mujoco.mj_forward(self.model, self.data)  # Ensure the new state is propagated

        # Set initial object position
        self.initial_object_pos = self.data.qpos[object_qpos_addr : object_qpos_addr + 3]
        self.initial_object_quat = self.data.qpos[object_qpos_addr + 3 : object_qpos_addr + 7]
        self.prev_object_pos = self.initial_object_pos.copy()
        self.prev_object_quat = self.initial_object_quat.copy()

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, 5)  # 1 is for forward dynamics

        obs = self._get_obs()
        done = self._get_done()
        reward = self._get_reward(done)
        info = {}

        return obs, reward, done, False, {}
    
    def _get_obs(self):
        robot_joint_pos = self.data.qpos[:10]
        # object_pos = self.data.qpos[10:13] # x,y,z
        # object_quat = self.data.qpos[13:17] # quaternion
        # target_pos = self.model.body("target").pos

        obs = robot_joint_pos.astype(np.float32)  # Convert to float32
        return obs

    def _get_done(self):
        return None
        current_object_pos = self.data.qpos[10:13] # x,y,z
        done = np.linalg.norm(current_object_pos - self.target_pos) < 0.05
        return done
    
    def _get_reward(self, done):
        reward = 0

        current_object_pos = self.data.qpos[8:11] # x,y,z
        current_object_quat = self.data.qpos[11:15] # quaternion

        # # Compute reward
        # current_distace = np.linalg.norm(current_object_pos - self.target_pos)
        # prev_distance = np.linalg.norm(self.prev_object_pos - self.target_pos)
        # initial_distance = np.linalg.norm(self.initial_object_pos - self.target_pos)
        # reward += (prev_distance - current_distace) / initial_distance

        # # Update previous object position
        # self.prev_object_pos = current_object_pos.copy()
        # self.prev_object_quat = current_object_quat.copy()

        if done:
            reward += 2
        return reward
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            return mujoco.mj_render(self.model, self.data, camera_name="top")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None