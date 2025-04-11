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
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Define action and observation space
        action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
        state_dim = action_dim + 3 # number of joints + EE pos
        image_dim = (2, 256, 256, 3) # 256x256x6 image (2xHxWxC, 0-1 RGB image)

        action_low = self.model.actuator_ctrlrange[:, 0].copy()
        action_high = self.model.actuator_ctrlrange[:, 1].copy()
        state_low = np.concatenate([action_low, [-np.inf]*3])
        state_high = np.concatenate([action_high, np.inf*np.ones(3)])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32)
        self.state_space = spaces.Box(low=state_low, high=state_high, shape=(state_dim,), dtype=np.float32)
        self.image_space = spaces.Box(low=0, high=1, shape=image_dim, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": self.state_space,
            "image": self.image_space
        })

        print("Action space:", self.action_space)
        print("Observation space:", self.observation_space)

        # Constants
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]]
        self.max_object_x = self.model.body("table").pos[0] + self.model.geom("table_geom").size[0]
        self.min_object_x = self.model.body("table").pos[0] - self.model.geom("table_geom").size[0]
        self.max_object_y = self.model.body("table").pos[1] + self.model.geom("table_geom").size[1]
        self.min_object_y = self.model.body("table").pos[1] - self.model.geom("table_geom").size[1]

        # Data for rewards and termination
        self.initial_object_pos = None
        self.prev_object_pos = None
        self.target_pos = None
        self.current_object_pos = None
        self.current_object_vel = None
        self.success = False
        self.out_of_bounds = False

        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self.top_renderer = None
        self.side_renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Reset MuJoCo model and data to home position
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe('home').qpos
        self.data.ctrl[:] = self.model.keyframe('home').ctrl

        # Randomly select object color
        object_color_name = self.np_random.choice(self.colors)
        object_color = self.color_map[object_color_name]
        self.model.geom("object_geom").rgba = object_color

        # Randomly offset object position
        # object_xy_delta_pos = self.np_random.uniform(-0.15, 0.15, size=(2,))
        object_delta_x_pos = self.np_random.uniform(0, 0.15)
        object_delta_y_pos = self.np_random.uniform(-0.15, 0.15)
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += (object_delta_x_pos, object_delta_y_pos)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = new_object_pos  

        mujoco.mj_forward(self.model, self.data)

        # Set pos for reward calculation
        self.target_pos = self.model.body(f"{object_color_name[0]}_target").pos.copy()
        self.initial_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy()
        self.prev_object_pos = self.initial_object_pos.copy()
        self.current_object_pos = self.initial_object_pos.copy()
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy()

        self.success = False
        self.out_of_bounds = False

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action + self.model.keyframe('home').ctrl # delta from home position
        # self.data.ctrl[:] += action # delta from current position
        mujoco.mj_step(self.model, self.data, 10) # 5 substeps

        # Update object info
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # x,y,z
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # dx,dy,dz

        # Check success and out of bounds
        self.success = self._check_success()
        self.out_of_bounds = self._check_out_of_bounds()
        # self.too_far = self._check_too_far()

        obs = self._get_obs()
        done = self._get_done()
        reward = self._get_reward()
        info = {}

        # Update previous object position
        self.prev_object_pos = self.current_object_pos.copy()

        # xmat = self.data.site_xmat[self.end_effector_id]  # shape: (9,)
        # quat = np.zeros(4)
        # mujoco.mju_mat2Quat(quat, xmat)
        # print(f"EE Xmat: {xmat}")
        # print(f"EE Quat: {quat}")

        return obs, reward, done, False, {}
    
    def _get_obs(self):
        robot_joint_angles = self.data.qpos[:7]
        end_effector_pos = self.data.site_xpos[self.end_effector_id]
        state = np.concatenate([robot_joint_angles, end_effector_pos])
        image = self._get_camera_image()
        obs = {"state": state, "image": image}
        # object_pos = self.data.qpos[7:10] # x,y,z
        # object_quat = self.data.qpos[10:14] # quaternion

        return obs

    def _get_done(self):
        done = self.success or self.out_of_bounds
        return done

    def _get_reward(self):
        '''
        Reward function:
        1. Distance to target (2 * prev_distance - current_distance)
        2. Success (1 if success, 0 otherwise) (TERMINAL)
        3. Out of bounds (-1 if out of bounds, 0 otherwise) (TERMINAL)
        4. Distance to end effector (penalty if too far) -0.005 * distance, max dist 0.4 (-0.002 * 500 = -1) 0.002 * 500
        5. Height penalty (penalty if too high or too low) -0.02 * abs(ee_z - 0.25), max height 0.45 (-0.02 * (0.45 - 0.25) * 500 = -2) 0.004 * 500
        6. Living penalty (-0.001)

        if stay still and close to target, max reward = -0.001 * 500 = -0.5
        anything below -0.5 is progress
        max negative = -2 -1 -0.5 = -3.5
        max positive = 1.2 + 1 = 2.2
        '''
        reward = 0

        # Compute ditance reward
        current_distace = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2])
        prev_distance = np.linalg.norm(self.prev_object_pos[:2] - self.target_pos[:2])
        initial_distance = np.linalg.norm(self.initial_object_pos[:2] - self.target_pos[:2])
        reward += 2 * (prev_distance - current_distace) # max 0.60 * 2 = 1.2
        # reward += 2 * (prev_distance - current_distace) / initial_distance # normalized distance reward and scale by x2

        if reward < 1e-7 and reward > -1e-7: # Too small reward
            reward = 0

        if self.success:
            reward += 1

        if self.out_of_bounds:
            reward -= 1

        # if self.too_far:
        #     reward -= 0.001

        x, y, z = self.current_object_pos
        ee_x, ee_y, ee_z = self.data.site_xpos[self.end_effector_id]
        distance = np.linalg.norm((x - ee_x, y - ee_y))

        # distance penalty
        if distance > 0.4:  # max is 500 * 0.005 * 0.4 = 1
            reward -= 0.005 * 0.4
        elif distance > 0.1: # 0.0707 corner + 0.01 radiues
            reward -= 0.005 * distance

        # height penalty 
        # more strict cos i want to keep ee in object level
        # but easy to learn beacause 0.25 is a fixed value
        if ee_z > 0.45: # max is 500 * 0.02 * 0.2 = 2
            reward -= 0.02 * (0.45 - 0.25)
        elif ee_z > 0.28 or ee_z < 0.22:
            reward -= 0.02 * abs(ee_z - 0.25)

        # Living penalty
        reward -= 0.001 # max is 500 * 0.001 = 0.5

        # # Height penalty
        # ee_x, ee_y, ee_z = self.data.site_xpos[self.end_effector_id]

        # if ee_z > 0.30 or ee_z < 0.2:
        #     reward -= 0.002

        return reward
    
    def _check_success(self):
        in_target = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2]) < 0.05
        still = np.linalg.norm(self.current_object_vel) < 0.01
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
        # max xy dist is 0.25, max z dist is 0.3 (block height), min z dist is 0.2 (table height)
        return distance > 0.15 or ee_z > 0.3 or ee_z < 0.2

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
        # elif self.render_mode == "camera":
        #     pass
        #     if self.viewer is None:
        #         self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        #     cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")
        #     if cam_id != -1:
        #         self.viewer.cam.fixedcamid = cam_id
        #         self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        #     self.viewer.sync()

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