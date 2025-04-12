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
change action space to x,y,z, open/close gripper
Change to render instead of viewer
'''

class PickPlaceCustomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path, render_mode="human"):
        super().__init__()
        print("Initializing PickPlaceCustomEnv...")
        self.np_random = None

        self.colors = ['red', 'green', 'blue']
        self.color_map = {
            'red': [1.0, 0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0]
        }

        # Load MuJoCo model and data
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Define action and observation space
        action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
        state_dim = action_dim + 3 # number of joints + EE pos
        image_dim = (2, 256, 256, 3) # 256x256x6 image (2xHxWxC, 0-1 RGB image)

        action_low = self.model.actuator_ctrlrange[:, 0].copy()
        action_high = self.model.actuator_ctrlrange[:, 1].copy()
        state_low = np.concatenate([action_low,-np.ones(3)])
        state_high = np.concatenate([action_high, np.ones(3)])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32)
        self.state_space = spaces.Box(low=state_low, high=state_high, shape=(state_dim,), dtype=np.float32)
        self.image_space = spaces.Box(low=0, high=1, shape=image_dim, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": self.state_space,
            "image": self.image_space
        })

        print("Action space:", self.action_space)
        print("Observation space:", self.observation_space)

        
        # Curriculum variables
        self.level = 0

        """
        Fail cond always active (OOB, too far)
        level 0: Learn to get ee close to object while keeping ee upright
            Random object color, Fixed object position, Penalty for non upright ee, not within level ee and far ee
        level 1: Learn to get ee close to object while keeping ee upright, but diff locations
            Random object color, Random object position, Penalty for non upright ee, not within level ee and far ee
        level 2: Learn to get ee close to object while keeping ee upright, but diff locations and push towards fixed front goal
            Random object color, Random object position, Fixed Goal with same color in front Penalty for non upright ee, not within level ee and far ee, Reward for pushing object towards goal, Success Term
        level 3: Learn to get ee close to object while keeping ee upright, but diff locations and push towards fixed 3 goal
            Random object color, Random object position, Fixed Goal with same color in front Penalty for non upright ee, not within level ee and far ee, Reward for pushing object towards goal, Success Term
        """

        # Constants
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.attachment_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]]
        self.max_object_x = self.model.body("table").pos[0] + self.model.geom("table_geom").size[0]
        self.min_object_x = self.model.body("table").pos[0] - self.model.geom("table_geom").size[0]
        self.max_object_y = self.model.body("table").pos[1] + self.model.geom("table_geom").size[1]
        self.min_object_y = self.model.body("table").pos[1] - self.model.geom("table_geom").size[1]

        # Data for rewards calculation
        self.initial_object_pos = None
        self.prev_object_pos = None
        self.target_pos = None
        self.current_object_pos = None
        self.current_object_vel = None

        # Termination flags
        self.success = False
        self.out_of_bounds = False
        self.too_far = False

        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self.top_renderer = None
        self.side_renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Curriculum
        if options is not None:
            self.level = options.get("level", 0)

        # Reset MuJoCo model and data to home position
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe('home').qpos
        self.data.ctrl[:] = self.model.keyframe('home').ctrl

        # Randomly select object color
        object_color_name = self.np_random.choice(self.colors)
        object_color = self.color_map[object_color_name]
        self.model.geom("object_geom").rgba = object_color

        # Randomly offset object position
        if self.level == 0: # Fixed object position
            object_delta_x_pos = 0.05
            object_delta_y_pos = 0.0
        else: # Random object position
            object_delta_x_pos = self.np_random.uniform(0, 0.15)
            object_delta_y_pos = self.np_random.uniform(-0.15, 0.15)
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += (object_delta_x_pos, object_delta_y_pos)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = new_object_pos
        
        # For target position
        if self.level in [0, 1]:
            for color in self.colors:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{color[0]}_target_geom")
                self.model.geom_rgba[geom_id][-1] = 0.0  # Set alpha to 0
        elif self.level == 2: # Fixed target position
            front_target_pos = self.model.body(f"g_target").pos.copy()
            same_color_target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{object_color_name[0]}_target")
            self.model.body_pos[same_color_target_id] = front_target_pos
        elif self.level == 3: # Random target position
            pass # default in file

        mujoco.mj_forward(self.model, self.data)

        # Set data for rewards calculation
        self.target_pos = self.model.body(f"{object_color_name[0]}_target").pos.copy()
        self.initial_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy()
        self.prev_object_pos = self.initial_object_pos.copy()
        self.current_object_pos = self.initial_object_pos.copy()
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy()

        # Set termination flags
        self.success = False
        self.out_of_bounds = False
        self.too_far = False

        return self._get_obs(), {}

    def step(self, action):
        # self.data.ctrl[:] = action # takes in absolute joint angles
        self.data.ctrl[:] += action # takes in relative joint angles
        # self.data.ctrl[:] = action + self.model.keyframe('home').ctrl # takes in delta joint angles from home position
        mujoco.mj_step(self.model, self.data, 10) # 10 substeps 2ms each = 20ms = 50Hz

        # Update object info
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # x,y,z
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # dx,dy,dz

        # Check success and out of bounds
        self.success = self._check_success()
        self.out_of_bounds = self._check_out_of_bounds()
        self.too_far = self._check_too_far()

        obs = self._get_obs()
        done = self._get_done()
        reward = self._get_reward()
        info = {}

        # Update previous object position
        self.prev_object_pos = self.current_object_pos.copy()

        return obs, reward, done, False, {}
    
    def _get_obs(self):
        robot_joint_angles = self.data.qpos[:7]
        end_effector_pos = self.data.site_xpos[self.end_effector_id]
        state = np.concatenate([robot_joint_angles, end_effector_pos]) # (7 + 3 = 10)
        image = self._get_camera_image()
        obs = {"state": state, "image": image}
        # object_pos = self.data.qpos[7:10] # x,y,z
        # object_quat = self.data.qpos[10:14] # quaternion

        return obs

    def _get_done(self):
        done = self.success or self.out_of_bounds or self.too_far
        return done

    def _get_reward(self):
        '''
        Reward function:
        '''
        reward = 0

        object_x, object_y, object_z = self.current_object_pos
        ee_x, ee_y, ee_z = self.data.site_xpos[self.end_effector_id]
        attachment_x, attachment_y, attachment_z = self.data.site_xpos[self.attachment_site_id]

        # upright penalty for ee, angle always positive 0 to 180 deg
        # if 90 degree, 90/180 * 0.005 * 300 = 0.75
        z_distance = attachment_z - ee_z # should be positive if ee is below, if z_distance < 0, then ee is above
        length = np.linalg.norm((object_x - attachment_x, object_y - attachment_y, object_z - attachment_z)) # NOTE fixed length of 0.1
        z_over_length = np.clip(z_distance / length, -1, 1) # clip to -1 to 1
        angle = np.arccos(z_over_length)
        # print(f"EE: {ee_x, ee_y, ee_z} | At: {attachment_x, attachment_y, attachment_z} | Angle: {angle/np.pi*180} | z_distance: {z_distance} | length: {length} | z_over_length: {z_over_length}")
        if angle > np.pi / 9: # if angle > 20 degrees
            reward -= 0.005 * angle / np.pi

        # distance penalty for ee_x, ee_y to be close to object_x, object_y
        # distance > 0.4 leads to termination and -1 penalty
        # max penalty = 0.005 * 0.35 * 300 = 0.525  
        distance = np.linalg.norm((object_x - ee_x, object_y - ee_y))
        if distance > 0.11: # NOTE 0.0707 corner of object + 0.01 radius of ee
            reward -= 0.005 * distance

        if self.level >= 2:
            # push distance reward
            # normalized distance to be / initial distance and scale by x2
            current_distace = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2])
            prev_distance = np.linalg.norm(self.prev_object_pos[:2] - self.target_pos[:2])
            initial_distance = np.linalg.norm(self.initial_object_pos[:2] - self.target_pos[:2])
            # reward += 2 * (prev_distance - current_distace)
            reward += 2 * (prev_distance - current_distace) / initial_distance
            if self.success:
                print("WE DID IT!")
                reward += 3

        # Living penalty
        # max penalty = 0.001 * 300 = -0.3
        # reward -= 0.001

        # Terimination rewards
        if self.out_of_bounds or self.too_far:
            reward -= 3

        return reward
    
    def _check_success(self):
        in_target = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2]) < 0.05 # NOTE 0.05 is 0.5 width of object
        still = np.linalg.norm(self.current_object_vel) < 0.01 # NOTE arbitrary speed threshold
        return in_target and still
    
    def _check_out_of_bounds(self):
        x, y, z = self.current_object_pos
        x_check = x > self.max_object_x or x < self.min_object_x
        y_check = y > self.max_object_y or y < self.min_object_y
        return x_check or y_check
    
    def _check_too_far(self):
        x, y, z = self.current_object_pos
        ee_x, ee_y, ee_z = self.data.site_xpos[self.end_effector_id]
        distance = np.linalg.norm((x - ee_x, y - ee_y))
        return distance > 0.35 or ee_z > 0.29 or ee_z < 0.21 # NOTE abitrary thresholds

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