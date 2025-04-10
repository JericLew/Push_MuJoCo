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
        # super().reset(seed=seed)
        # Reset MuJoCo model and data to home position
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe('home').qpos
        self.data.ctrl[:] = self.model.keyframe('home').ctrl

        # Randomly select object color
        object_color_name = np.random.choice(self.colors)
        object_color = self.color_map[object_color_name]
        self.model.geom("object_geom").rgba = object_color

        # Randomly offset object position
        object_xy_delta_pos = np.random.uniform(-0.15, 0.15, size=(2,))
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += object_xy_delta_pos
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
        self.data.ctrl[:] = action # + self.model.keyframe('home').ctrl # delta from home position
        mujoco.mj_step(self.model, self.data, 5) # 5 substeps

        # Update object info
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # x,y,z
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # dx,dy,dz

        # Check success and out of bounds
        self.success = self._check_success()
        self.out_of_bounds = self._check_out_of_bounds()
        self.too_far = self._check_too_far()

        obs = self._get_obs()
        done = self._get_done()
        reward = self._get_reward(done)
        info = {}

        # Update previous object position
        self.prev_object_pos = self.current_object_pos.copy()

        return obs, reward, done, False, {}
    
    def _get_obs(self):
        robot_joint_angles = self.data.qpos[:7]
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        end_effector_pos = self.data.site_xpos[end_effector_id]
        # end_effector_q
        state = np.concatenate([robot_joint_angles, end_effector_pos])
        image = self._get_camera_image()
        obs = {"state": state, "image": image}
        # object_pos = self.data.qpos[7:10] # x,y,z
        # object_quat = self.data.qpos[10:14] # quaternion

        return obs

    def _get_done(self):
        done = self.success or self.out_of_bounds or self.too_far
        return done
    
    def _get_reward(self, done):
        '''
        Reward function:
        - Distance reward: 2 * (prev_distance - current_distance) / initial_distance
        - Success reward: +2 if success
        - Out of bounds penalty: -2 if out of bounds

        Total reward if success = 4
        '''
        reward = 0

        # Compute ditance reward
        current_distace = np.linalg.norm(self.current_object_pos[:2] - self.target_pos[:2])
        prev_distance = np.linalg.norm(self.prev_object_pos[:2] - self.target_pos[:2])
        initial_distance = np.linalg.norm(self.initial_object_pos[:2] - self.target_pos[:2])
        reward += 2 * (prev_distance - current_distace) / initial_distance # normalized distance reward and scale by x2

        if reward < 1e-7 and reward > -1e-7: # Too small reward
            reward = 0

        if self.success:
            reward += 2

        if self.out_of_bounds or self.too_far:
            reward -= 2

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
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        ee_x, ee_y, ee_z = self.data.site_xpos[end_effector_id]
        distance = np.linalg.norm((x - ee_x, y - ee_y, z - ee_z))
        return distance > 0.6

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