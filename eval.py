import os
import torch
import numpy as np

import gymnasium as gym

from model.common.cnn import ImageEncoder, DualImageEncoder
from model.push_actor import PushNNActor, PushNNPrivilegedActor
from model.push_critic import PushNNCritic, PushNNPrivilegedCritic


## Eval Hyperparameters
eval_runs = 100

## Environment Hyperparameters
action_type = "delta_xy" # delta_xy, delta_angle, absolute_angle
privileged = True # Train with privileged information?
random_object_pos = True # Randomize object position?
max_episode_steps = 300

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = f"saved_pth/{'privileged_' if privileged else ''}actor.pth"
full_weights_path = os.path.join(script_dir, weights_path)

## Network Hyperparameters
if action_type == "delta_xy":
    fixed_std = 0.015
    learn_fixed_std = True
    std_min = 0.0025
    std_max = 0.025
elif action_type == "delta_angle":
    fixed_std = 0.03
    learn_fixed_std = True
    std_min = 0.005
    std_max = 0.05
elif action_type == "absolute_angle":
    fixed_std = 0.03
    learn_fixed_std = True
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
                action_type=action_type,
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
action_low = env.action_space.low
action_high = env.action_space.high
print(f"State Space: {env.observation_space['state']}")
print(f"Image Space: {env.observation_space['image']}")
print(f"Privileged Space: {env.observation_space['privileged']}")
print(f"Action Space: {env.action_space}")
print(f"Action Low: {action_low}, Action High: {action_high}")

if privileged: 
    actor = PushNNPrivilegedActor(
        action_low=action_low,
        action_high=action_high,
        privileged_dim=privileged_dim,
        action_dim=action_dim,
        mlp_dims=[1024, 1024, 1024, 1024],
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
        action_low=action_low,
        action_high=action_high,
        state_dim=state_dim,
        action_dim=action_dim,
        mlp_dims=[1024, 1024, 1024, 1024],
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
actor.load_state_dict(torch.load(full_weights_path))
actor.eval()  # Set the model to evaluation mode
actor.to("cuda")  # Move the model to GPU

n_success = 0
n_out_of_bounds = 0
n_too_far = 0
n_wrong_target = 0
n_truncated = 0
rewards = []

for run in range(eval_runs):
    print(f"Start Run: {run}")
    episode_reward = 0
    obs, info = env.reset()

    for step in range(max_episode_steps):
        obs = {key: torch.from_numpy(obs[key]).to("cuda").float().unsqueeze(0) for key in obs.keys()}
        obs["image"] = obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
        with torch.no_grad():
            dist = actor.get_distribution(obs)
        action = dist.mean # deterministic action
        action = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)

        n_success += info["success"]
        n_out_of_bounds += info["out_of_bounds"]
        n_too_far += info["too_far"]
        n_wrong_target += info["wrong_target"]
        n_truncated += truncated

        episode_reward += reward

        env.render()

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps with reward: {episode_reward}")
            rewards.append(episode_reward)
            break
    print(f"Run: {run} | Episode Reward: {episode_reward} | Success: {info['success']} | Out of Bounds: {info['out_of_bounds']} | Too Far: {info['too_far']} | Wrong Target: {info['wrong_target']}")
    print(f"=="*20)
env.close()

success_rate = n_success / eval_runs
out_of_bounds_rate = n_out_of_bounds / eval_runs
too_far_rate = n_too_far / eval_runs
wrong_target_rate = n_wrong_target / eval_runs
truncated_rate = n_truncated / eval_runs
failure_rate = (n_out_of_bounds + n_too_far + n_wrong_target + n_truncated) / eval_runs
print(f"Success Rate: {success_rate} | Out of Bounds Rate: {out_of_bounds_rate} | Too Far Rate: {too_far_rate} | Wrong Target Rate: {wrong_target_rate} | Truncated Rate: {truncated_rate} | Failure Rate: {failure_rate}")
print(f"Mean Episode Reward: {np.mean(rewards)} | Std Episode Reward: {np.std(rewards)}")