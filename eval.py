import os
import torch
import numpy as np

import gymnasium as gym

from model.common.cnn import ImageEncoder, DualImageEncoder
from model.push_actor import PushNNActor, PushNNPrivilegedActor
from model.push_critic import PushNNCritic, PushNNPrivilegedCritic


## Eval Hyperparameters
max_episode_steps = 300
eval_steps = 9000

## Environment Hyperparameters
privileged = True
random_object_pos = False

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
print(f"State Space: {env.observation_space['state']}")
print(f"Image Space: {env.observation_space['image']}")
print(f"Privileged Space: {env.observation_space['privileged']}")
print(f"Action Space: {env.action_space}")

# NOTE: action is delta x, y pos (magnitude of 0.05 / 10*2(10^-3) seconds = 0.5 m/s)
privileged_action_dim = 2 # delta x, delta y
privileged_action_high = np.ones(privileged_action_dim) * 0.05
privileged_action_low = np.ones(privileged_action_dim) * (-0.05)

# NOTE: action is delta joint angles (magnitude of 3 degrees / 10*2(10^-3) seconds = 150 degrees/s)
action_dim = 7 # 7 delta joint angles
action_high = np.ones(action_dim) * (3 / 180 * np.pi)
action_low = np.ones(action_dim) * (-3 / 180 * np.pi)
# action_high = venv.action_space.high # NOTE: for absolute angles action
# action_low = venv.action_space.low

if privileged: 
    ## Network Hyperparameters
    fixed_std = 0.03
    learn_fixed_std = True
    std_min = 0.005
    std_max = 0.05
    
    actor = PushNNPrivilegedActor(
        action_low=privileged_action_low,
        action_high=privileged_action_high,
        privileged_dim=privileged_dim,
        action_dim=privileged_action_dim,
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
    ## Network Hyperparameters
    fixed_std = 0.03
    learn_fixed_std = True
    std_min = 0.005
    std_max = 0.05

    image_encoder_actor = DualImageEncoder(
        image_input_shape=image_dim,
        feature_dim=256,
    )
    actor = PushNNActor(
        backbone=image_encoder_actor,
        action_low=action_low,
        action_high=action_high,
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
weights_path = os.path.expanduser("~/ME5406/project_2_code/saved_pth/privileged_actor.pth") 
actor.load_state_dict(torch.load(weights_path))
actor.eval()  # Set the model to evaluation mode
actor.to("cuda")  # Move the model to GPU

obs, info = env.reset()
print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")

for step in range(eval_steps):
    print(f"Step: {step}")
    print(f"Obs: {obs['state'].shape}, {obs['image'].shape}")
    obs = {key: torch.from_numpy(obs[key]).to("cuda").float().unsqueeze(0) for key in obs.keys()}  # Convert to torch tensors
    obs["image"] = obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
    print(f"Obs: {obs['state'].shape}, {obs['image'].shape}")
    with torch.no_grad():
        dist = actor.get_distribution(obs)
    action = dist.mean # deterministic action
    
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
