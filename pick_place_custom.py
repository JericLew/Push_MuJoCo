import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class CustomPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode="human"):
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "franka_emika_panda/scene_pick_and_place.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Define action space (x, y, z movement + gripper open/close)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Define observation space (robot + object + target positions)
        obs_dim = 9  # Example: (gripper xyz, object xyz, target xyz)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self.viewer = None

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = self.model.keyframe('home').qpos
        self.data.ctrl = self.model.keyframe('home').ctrl
        mujoco.mj_forward(self.model, self.data)  # Ensure the new state is propagated

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)  # Ensure valid actions
        # self.data.qpos[:3] += action[:3] * 0.01  # Apply action safely
        mujoco.mj_step(self.model, self.data)

        # Compute reward
        object_pos = self.data.qpos[3:6]
        target_pos = np.array([0.3, 0.3, 0.1], dtype=np.float32)
        reward = -np.linalg.norm(object_pos - target_pos)

        done = reward > -0.05

        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        gripper_pos = self.data.qpos[:3]
        object_pos = self.data.qpos[3:6]
        target_pos = np.array([0.3, 0.3, 0.1], dtype=np.float32)  # Ensure float32

        obs = np.concatenate([gripper_pos, object_pos, target_pos]).astype(np.float32)  # Convert to float32
        return obs
    
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

    
    # def reset(self, seed=None, options=None):
    #     mujoco.mj_resetData(self.model, self.data)
    #     self.data.qpos[:3] = np.array([0.2, 0.2, 0.2])  # Initialize gripper position
    #     self.data.qpos[3:6] = np.array([0.2, 0.2, 0.1])  # Initialize object position
    #     return self._get_obs(), {}

    # def step(self, action):
    #     # Apply action (move gripper)
    #     self.data.qpos[:3] += action[:3] * 0.01  # Move gripper
    #     mujoco.mj_step(self.model, self.data)

    #     # Compute reward (if object is near target)
    #     object_pos = self.data.qpos[3:6]
    #     target_pos = np.array([0.3, 0.3, 0.1])
    #     reward = -np.linalg.norm(object_pos - target_pos)

    #     done = reward > -0.05  # Task success if object is close to target

    #     return self._get_obs(), reward, done, False, {}

    # def _get_obs(self):
    #     gripper_pos = self.data.qpos[:3]
    #     object_pos = self.data.qpos[3:6]
    #     target_pos = np.array([0.3, 0.3, 0.1])
    #     return np.concatenate([gripper_pos, object_pos, target_pos])