import os
import torch
import numpy as np

import gymnasium as gym

from model.common.cnn import ImageEncoder, DualImageEncoder
from model.push_actor import PushNNActor, PushNNPrivilegedActor
from model.push_critic import PushNNCritic, PushNNPrivilegedCritic


## Eval Hyperparameters
max_episode_steps = 300

## Environment Hyperparameters
privileged = True
random_object_pos = False

if privileged: # NOTE: action is delta x, y pos
    action_high = 0.05
    action_low = -0.05
else: # NOTE: action is delta joint angles
    action_high = 3 / 180 * np.pi # 3 degrees in radians / 10*2(10^-3) seconds = 150 degrees/s
    action_low = -3 / 180 * np.pi # 3 degrees in radians / 10*2(10^-3) seconds = 150 degrees/s

## Network Hyperparameters
if privileged:
    fixed_std = 0.03
    learn_fixed_std = True # True
    std_min = 0.005
    std_max = 0.05
else:
    fixed_std = 0.03
    learn_fixed_std = True # True
    std_min = 0.005
    std_max = 0.05

## Load Push Environment
gym.register(
    id="PickPlaceCustomEnv-v0",
    entry_point="push_custom:PickPlaceCustomEnv",
    max_episode_steps=max_episode_steps,
)
xml_path = "franka_emika_panda/scene_push.xml"

env = gym.make("PickPlaceCustomEnv-v0",
                xml_path=xml_path,
                privileged=privileged,
                random_object_pos=random_object_pos,
                render_mode="human",
                max_episode_steps=max_episode_steps,
                )
state_dim = env.observation_space["state"].shape[0]
image_dim = env.observation_space["image"].shape[1:] # (B, N, H, W, C) -> (H, W, C)
image_dim = (image_dim[2], image_dim[0], image_dim[1]) # (H, W, C) -> (C, H, W)
privileged_dim = env.observation_space["privileged"].shape[0]
action_dim = env.action_space.shape[0]

print("State Space: ", state_dim)
print("Image Space: ", image_dim)
print("Action Space: ", action_dim)

obs, info = env.reset()
print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")

## Load Model Weights
actor = None
if privileged:
        actor = PushNNPrivilegedActor(
        privileged_dim=privileged_dim,
        action_dim=action_dim,
        mlp_dims=[512, 512, 512, 512],
        activation_type="Mish",
        tanh_output=True,
        residual_style=True,
        use_layernorm=False,
        dropout=0.0,
        fixed_std=fixed_std,
        learn_fixed_std=learn_fixed_std,
        std_min=std_min,
        std_max=std_max,
    )
else:
    image_encoder_actor = DualImageEncoder(
        image_input_shape=image_dim,
        feature_dim=256,
    )
    actor = PushNNActor(
        backbone=image_encoder_actor,
        state_dim=state_dim,
        action_dim=action_dim,
        mlp_dims=[512, 512, 512, 512],
        activation_type="Mish",
        tanh_output=True,
        residual_style=True,
        use_layernorm=False,
        dropout=0.0,
        fixed_std=fixed_std,
        learn_fixed_std=learn_fixed_std,
        std_min=std_min,
        std_max=std_max,
        visual_feature_dim=128,
    )
weights_path = os.path.expanduser("~/Jeric/Push_MuJoCo/saved_weights/privileged_actor.pth") 
actor.load_state_dict(torch.load(weights_path))
actor.eval()  # Set the model to evaluation mode
actor.to("cuda")  # Move the model to GPU

for step in range(max_episode_steps):
    print(f"Step: {step}")
    print(f"Obs: {obs['state'].shape}, {obs['image'].shape}")
    obs = {key: torch.from_numpy(obs[key]).to("cuda").float().unsqueeze(0) for key in obs.keys()}  # Convert to torch tensors
    obs["image"] = obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
    print(f"Obs: {obs['state'].shape}, {obs['image'].shape}")
    with torch.no_grad():
        mean, std = actor.forward(obs)
    action = mean # deterministic action

    action = action.squeeze(0).cpu().numpy()  # Convert to numpy array
    print(f"Action: {action}")

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    print(f"Joint Angles: {obs['state'][:7]}")
    print(f"EE Pose: {obs['state'][7:]}")
    print(f"Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
    if terminated or truncated:
        obs, info = env.reset()
    print(f"=="*20)

env.close()
