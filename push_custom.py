import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

import torch
import inverse_kinematics as ik

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
'''

class PickPlaceCustomEnv(gym.Env):
    """
    Custom Pick and Place Environment using MuJoCo.
    This environment simulates a robotic arm to push an object to a target position.
    The robot arm is a Franka Emika Panda with 7 degrees of freedom.
    The object is a coloured cube that can be pushed to the same fixed coloured target position.

    The environment supports both privileged and non-privileged action spaces.
    Privileged action spaec: delta x, delta y (NOTE z is fixed to object height)
    Non-privileged action space: delta x, delta y
    """
    metadata = {"render_modes": ["human"], "render_fps": 20, "action_types": ["delta_xy", "delta_angle", "absolute_angle"]}

    def __init__(self, xml_path, action_type="delta_xy", privileged=False, random_object_pos=False, render_mode="human"):
        super().__init__()
        print("Initializing PickPlaceCustomEnv...")
        self.np_random = None

        self.colors = ['red', 'green', 'blue']
        self.color_map = {
            'red': [1.0, 0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0]
        }

        ## Load MuJoCo model and data
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        ## Option Flags
        self.action_type = action_type
        self.privileged = privileged
        self.random_object_pos = random_object_pos

        ## Define action and observation space
        if action_type == "delta_xy": # 0.02 / (25*2(10^-3)) seconds = 1 m/s = 100 cm/s)
            action_dim = 2
            action_low = np.ones(action_dim) * (-0.02)
            action_high = np.ones(action_dim) * (0.02)
        elif action_type == "delta_angle": # 7.5 degrees / (25*2(10^-3)) seconds = 150 degrees/s
            action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
            action_low = np.ones(action_dim) * (-7.5 / 180 * np.pi)
            action_high = np.ones(action_dim) * (7.5 / 180 * np.pi)
        elif action_type == "absolute_angle":
            action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
            action_low = self.model.actuator_ctrlrange[:, 0].copy()
            action_high = self.model.actuator_ctrlrange[:, 1].copy()
        else:
            raise ValueError(f"Invalid action type: {action_type}")
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32)

        state_dim = self.model.nu + 3 # number of joints + EE pos
        image_dim = (2, 256, 256, 3) # 256x256x6 image (2xHxWxC, 0-1 RGB image)
        privileged_dim = 13 # ee pos (3) + object pos (3) + target pos (3) + object quat (4)
        state_low = np.concatenate([self.model.actuator_ctrlrange[:, 0].copy(), -np.ones(3)])
        state_high = np.concatenate([self.model.actuator_ctrlrange[:, 1].copy(), np.ones(3)])
        privileged_low = np.concatenate([-np.ones(13)])
        privileged_high = np.concatenate([np.ones(13)])
        self.state_space = spaces.Box(low=state_low, high=state_high, shape=(state_dim,), dtype=np.float32)
        self.image_space = spaces.Box(low=0, high=1, shape=image_dim, dtype=np.float32)
        self.privileged_space = spaces.Box(low=privileged_low, high=privileged_high, shape=(privileged_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": self.state_space,
            "image": self.image_space,
            "privileged": self.privileged_space,
        })

        ## Constants
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.attachment_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]]
        self.max_object_x = self.model.body("table").pos[0] + self.model.geom("table_geom").size[0]
        self.min_object_x = self.model.body("table").pos[0] - self.model.geom("table_geom").size[0]
        self.max_object_y = self.model.body("table").pos[1] + self.model.geom("table_geom").size[1]
        self.min_object_y = self.model.body("table").pos[1] - self.model.geom("table_geom").size[1]

        ## Data for rewards calculation
        self.initial_object_pos = None
        self.prev_object_pos = None
        self.target_pos = None
        self.current_object_pos = None
        self.current_object_vel = None

        ## Termination flags
        self.success = False
        self.out_of_bounds = False
        self.too_far = False
        self.wrong_target = False

        ## Rendering
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self.top_renderer = None
        self.side_renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        ## Options
        if options is not None:
            self.privileged = options.get("privileged", self.privileged)
            self.random_object_pos = options.get("random_object_pos", self.random_object_pos)

        ## Reset MuJoCo model and data to home position
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe('home').qpos
        self.data.ctrl[:] = self.model.keyframe('home').ctrl

        ## Randomly select object color
        object_color_name = self.np_random.choice(self.colors)
        object_color = self.color_map[object_color_name]
        self.model.geom("object_geom").rgba = object_color

        ## Set object position
        if self.random_object_pos: # Random object position
            object_delta_x_pos = self.np_random.uniform(0, 0.15)
            object_delta_y_pos = self.np_random.uniform(-0.15, 0.15)
        else: # Fixed object position
            object_delta_x_pos = 0.05
            object_delta_y_pos = 0.0
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += (object_delta_x_pos, object_delta_y_pos)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = new_object_pos

        mujoco.mj_forward(self.model, self.data)

        ## Set data for rewards calculation
        self.target_pos = self.model.body(f"{object_color_name[0]}_target").pos.copy()
        self.initial_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy()
        self.prev_object_pos = self.initial_object_pos.copy()
        self.current_object_pos = self.initial_object_pos.copy()
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy()

        ## Set termination flags
        self.success = False
        self.out_of_bounds = False
        self.too_far = False
        self.wrong_target = False

        obs = self._get_obs()

        ## Prep imitation info
        info = dict()
        info["success"] = self.success
        info["out_of_bounds"] = self.out_of_bounds
        info["too_far"] = self.too_far
        info["wrong_target"] = self.wrong_target

        return obs, info

    def step(self, action):
        # self.data.ctrl[:] = action # takes in absolute joint angles
        # self.data.ctrl[:] += action # takes in delta joint angles
        # self.data.ctrl[:] = action + self.model.keyframe('home').ctrl # takes in delta joint angles from home position

        ## Handle action
        target_pos = None
        if self.action_type == "delta_xy":
            ee_x, ee_y, ee_z = self.data.site_xpos[self.end_effector_id].copy() # x,y,z
            ee_x += action[0]
            ee_y += action[1]
            ee_z = self.data.qpos[9] # object z position
            target_pos = np.array([ee_x, ee_y, ee_z])
            target_quat = np.array([0, 0.7071068, -0.7071068, 0]) # EE straight up
            joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            ikresults = ik.qpos_from_site_pose(mjmodel=self.model, mjdata=self.data, site_name="end_effector", target_pos=target_pos, target_quat=target_quat, joint_names=joint_names)
            action = ikresults[0][:7]
            self.data.ctrl[:] = action
        elif self.action_type == "delta_angle":
            self.data.ctrl[:] += action
        elif self.action_type == "absolute_angle":
            self.data.ctrl[:] = action
        else:
            raise ValueError(f"Invalid action type: {self.action_type}")
        
        ## Step simulation
        mujoco.mj_step(self.model, self.data, 25) # 25 substeps 2ms each = 50ms = 20Hz

        ## Update object info
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # x,y,z
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # dx,dy,dz

        ## Check termination conditions
        self.success = self._check_success()
        self.out_of_bounds = self._check_out_of_bounds()
        self.too_far = self._check_too_far()
        self.wrong_target = self._check_wrong_target()

        obs = self._get_obs()
        done = self._get_done()
        reward = self._get_reward()

        ## Prep imitation info
        info = dict()
        info["success"] = self.success
        info["out_of_bounds"] = self.out_of_bounds
        info["too_far"] = self.too_far
        info["wrong_target"] = self.wrong_target

        ## Update previous object position
        self.prev_object_pos = self.current_object_pos.copy()

        return obs, reward, done, False, info
    
    def _get_obs(self):
        robot_joint_angles = self.data.qpos[:7].copy().astype(np.float32) # 7 joint angles
        end_effector_pos = self.data.site_xpos[self.end_effector_id].copy().astype(np.float32) # 3 end effector position
        state = np.concatenate([robot_joint_angles, end_effector_pos]).astype(np.float32) # (7 + 3 = 10)
        image = self._get_camera_image()

        ## Privileged information
        object_pos = self.data.qpos[7:10].copy().astype(np.float32) # object position
        object_quat = self.data.qpos[10:14].copy().astype(np.float32) # object quaternion
        privileged = np.concatenate([end_effector_pos, object_pos, self.target_pos, object_quat]).astype(np.float32) # (3 + 3 + 3 + 4 = 13)
        # object_qvel = self.data.qvel[7:14] # object velocity
        
        obs = {"state": state, "image": image, "privileged": privileged}
        return obs
    
    def _get_done(self):
        done = self.success or self.out_of_bounds or self.wrong_target or self.too_far
        return done

    def _get_reward(self):
        '''
        Reward function:
        '''
        reward = 0

        ee_pos = self.data.site_xpos[self.end_effector_id].copy()

        ## Object Distance Penalty
        object_distance = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2])
        initial_distance = np.linalg.norm(self.initial_object_pos[:2] - self.target_pos[:2])
        reward -= 0.005 * (object_distance / initial_distance)

        ## Terimination rewards
        if self.success:
            reward += 5
            print(f"WE DID IT! Object: {self.current_object_pos}, EE: {ee_pos} -> {self.target_pos}")
        if self.out_of_bounds:
            reward -= 5
            print(f"Out of bounds! Object: {self.current_object_pos}, EE: {ee_pos} -> {self.target_pos}")
        if self.too_far:
            reward -= 5
            print(f"Too far! Object: {self.current_object_pos}, EE: {ee_pos} -> {self.target_pos}")
        if self.wrong_target:
            reward -= 1.5
            print(f"Wrong target! Object: {self.current_object_pos}, EE: {ee_pos} -> {self.target_pos}")
        return reward
    
    def _check_success(self):
        in_target = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2]) < 0.10
        still = np.linalg.norm(self.current_object_vel) < 0.01 # NOTE arbitrary speed threshold
        return in_target # and still
    
    def _check_out_of_bounds(self):
        x, y, z = self.current_object_pos
        x_check = x > self.max_object_x or x < self.min_object_x
        y_check = y > self.max_object_y or y < self.min_object_y
        return x_check or y_check
    
    def _check_too_far(self):
        x, y, z = self.current_object_pos
        ee_x, ee_y, ee_z = self.data.site_xpos[self.end_effector_id]
        distance = np.linalg.norm((x - ee_x, y - ee_y))
        return distance > 0.35 # or ee_z > 0.29 or ee_z < 0.21 # NOTE abitrary thresholds
    
    def _check_wrong_target(self):
        for color in self.colors:
            color_target_pos = self.model.body(f"{color[0]}_target").pos.copy()
            if color_target_pos[0] == self.target_pos[0] and color_target_pos[1] == self.target_pos[1]:
                continue # if correct target, skip
            else:
                in_wrong_target = np.linalg.norm(self.current_object_pos[:2] - color_target_pos[:2]) < 0.10
                still = np.linalg.norm(self.current_object_vel) < 0.01 # NOTE arbitrary speed threshold
                if in_wrong_target: # and still:
                    return True
        return False

    def _get_camera_image(self, width=256, height=256):
        if self.top_renderer is None:
            self.top_renderer = mujoco.Renderer(self.model, height, width)
        if self.side_renderer is None:
            self.side_renderer = mujoco.Renderer(self.model, height, width)
        self.top_renderer.update_scene(self.data, camera="top_cam")
        top_image = self.top_renderer.render()
        self.side_renderer.update_scene(self.data, camera="side_cam")
        side_image = self.side_renderer.render()
        stacked_image = np.stack([top_image, side_image], axis=0) # (2, H, W, C)
        stacked_image = stacked_image.astype(np.float32) / 255.0 # Convert to float and normalize to [0, 1]
        return stacked_image
        
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        if self.top_renderer:
            self.top_renderer.close()
            self.top_renderer = None
        if self.side_renderer:
            self.side_renderer.close()
            self.side_renderer = None