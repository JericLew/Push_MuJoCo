import torch
from torch.distributions import Normal, Independent

import gymnasium as gym
from push_nn import PushNN

import os
from PIL import Image


gym.register(
    id="PickPlaceCustomEnv-v0",
    entry_point="push_custom:PickPlaceCustomEnv",
    max_episode_steps=1000,
)

xml_path = "franka_emika_panda/scene_push.xml"
venv = gym.make_vec("PickPlaceCustomEnv-v0",
                    xml_path=xml_path,
                    render_mode="human",
                    max_episode_steps=1000,
                    num_envs=2,
                    vectorization_mode="sync",
                    )
obs, info = venv.reset()
print(f"Obs: State: {obs['state'].shape}, Image: {obs['image'].shape}")

push_nn = PushNN()

for step in range(1000):
    print(f"Step: {step}")
    print(f"Obs: {obs['state'].shape}, {obs['image'].shape}")
    obs = {key: torch.from_numpy(obs[key]).to("cuda").float() for key in obs.keys()}  # Convert to torch tensors
    obs["image"] = obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
    print(f"Obs: {obs['state'].shape}, {obs['image'].shape}")
    (mean, std), value = push_nn.forward(obs)
    dist = Independent(Normal(mean, std), 1)  # 1 = number of reinterpreted batch dims
    action = dist.sample()  # shape: (batch_size, 7)

    mean = mean.detach().cpu().numpy()  # Convert to numpy array
    std = std.detach().cpu().numpy()  # Convert to numpy array
    value = value.detach().cpu().numpy()  # Convert to numpy array
    action = action.cpu().numpy()  # Convert to numpy array
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Value: {value}")
    print(f"Action: {action}")

    obs, reward, terminated, truncated, info = venv.step(action)

    # Save images to a folder with numbered names
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)

    for env_id, images in enumerate(obs["image"]):  # Iterate over batch of images
        for image_id, sub_img in enumerate(images):  # Iterate over N images in the batch
            img_pil = Image.fromarray((sub_img * 255).astype("uint8"))  # Convert to PIL Image
            img_pil.save(os.path.join(save_dir, f"env_{env_id}_img_{image_id}_step_{step}_.png"))
    # venv.render()
    # if reward > 1e-3 or reward < -1e-3:
    print(f"EE Pose: {obs['state'][7:]}")
    print(f"Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
    # if terminated or truncated:
    #     obs, info = venv.reset()
    print(f"=="*20)

venv.close()
