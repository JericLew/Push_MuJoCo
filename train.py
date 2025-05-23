import os
import torch
import numpy as np
import gymnasium as gym

from PPO import PPOAgent
from model.common.cnn import ImageEncoder, DualImageEncoder
from model.push_actor import PushNNActor, PushNNPrivilegedActor
from model.push_critic import PushNNCritic, PushNNPrivilegedCritic

if __name__ == "__main__":
    ## Environment Hyperparameters
    action_type = "delta_xy" # delta_xy, delta_angle, absolute_angle
    privileged = False # Train with privileged information?
    random_object_pos = True # Randomize object position?
    max_episode_steps = 300
    n_envs = 30
    vectorization_mode = "async"

    ## PPO Hyperparameters
    n_iterations = 10000

    if privileged: # Privileged PPO Hyperparameters
        use_wandb = True
        name = "privileged-delta_xy-random"
        save_model_interval = 25
        save_image_interval = 25
        batch_size = 300
        grad_accumulation_steps = 10
        n_updates_per_iteration = 10
        gamma = 0.999
        gae_lambda = 0.95
        vf_coef = 0.01
        bc_loss_coef = 0.0
        entropy_coef = 0.0 # 1e-5
        entropy_coef_decay = 0.95
        clip = 0.2
        actor_lr = 3e-4
        critic_lr = 5e-4
    else: # Non-Privileged PPO Hyperparameters
        use_wandb = True
        name = "imitation-delta_xy-random"
        save_model_interval = 25
        save_image_interval = 25
        batch_size = 200
        grad_accumulation_steps = 15
        n_updates_per_iteration = 10
        gamma = 0.999
        gae_lambda = 0.95
        vf_coef = 0.01
        bc_loss_coef = 1e-1
        bc_loss_coef_min = 1e-2
        bc_loss_coef_decay = 0.99
        entropy_coef = 1e-4
        entropy_coef_decay = 0.99
        clip = 0.2
        actor_lr = 3e-4
        critic_lr = 5e-4
        expert_path = os.path.expanduser("~/Jeric/Push_MuJoCo/saved_pth/privileged_actor.pth")

    ## Network Hyperparameters
    if action_type == "delta_xy":
        fixed_std = 0.015
        learn_fixed_std = True
        std_min = 0.01
        std_max = 0.02
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

    ## Make Environment
    gym.register(
        id="PickPlaceCustomEnv-v0",
        entry_point="push_custom:PickPlaceCustomEnv",
        max_episode_steps=max_episode_steps,
    )

    xml_path = "franka_emika_panda/scene_push.xml"
    venv = gym.make_vec("PickPlaceCustomEnv-v0",
                        xml_path=xml_path,
                        action_type=action_type,
                        privileged=privileged,
                        random_object_pos=random_object_pos,
                        render_mode=None,
                        max_episode_steps=max_episode_steps,
                        num_envs=n_envs,
                        vectorization_mode=vectorization_mode,
                        )

    ## Handle Observation and Action Space
    state_dim = venv.observation_space["state"].shape[1]
    image_dim = venv.observation_space["image"].shape[2:] # (B, N, H, W, C) -> (H, W, C)
    image_dim = (image_dim[2], image_dim[0], image_dim[1]) # (H, W, C) -> (C, H, W)
    privileged_dim = venv.observation_space["privileged"].shape[1]
    action_dim = venv.action_space.shape[1]
    action_low = venv.action_space.low[0]
    action_high = venv.action_space.high[0]
    print(f"State Space: {venv.observation_space['state']}")
    print(f"Image Space: {venv.observation_space['image']}")
    print(f"Privileged Space: {venv.observation_space['privileged']}")
    print(f"Action Space: {venv.action_space}")
    print(f"Action Low: {action_low}, Action High: {action_high}")
        
    ## Handle Privileged or Non-Privileged Training, if not privileged use imitation learning
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
        critic = PushNNPrivilegedCritic(
            privileged_dim=privileged_dim,
            mlp_dims=[512, 512, 512],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
            dropout=0.0,
        )
        expert_actor = None # No imitation learning
        
        ## Continue from checkpoint
        # base_dir = os.path.expanduser("~/Jeric/Push_MuJoCo/log/privileged-delta_xy-random/weights/")
        # pactor_pth_path = os.path.join(base_dir, "550/actor.pth")
        # actor.load_state_dict(torch.load(pactor_pth_path))
        # pcritic_pth_path = os.path.join(base_dir, "550/critic.pth")
        # critic.load_state_dict(torch.load(pcritic_pth_path))
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
        critic = PushNNPrivilegedCritic(
            privileged_dim=privileged_dim,
            mlp_dims=[512, 512, 512],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
            dropout=0.0,
        )
        expert_actor = PushNNPrivilegedActor(
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
        expert_actor.load_state_dict(torch.load(expert_path))
    
    ## Initialize PPO Agent
    rl_agent = PPOAgent(venv, actor=actor, critic=critic, expert_actor=expert_actor)
    rl_agent.init_hyperparameters(use_wandb=use_wandb,
                                    name=name,
                                    save_model_interval=save_model_interval,
                                    save_image_interval=save_image_interval,
                                    n_steps=max_episode_steps,
                                    n_envs=n_envs,
                                    batch_size=batch_size,
                                    grad_accumulation_steps=grad_accumulation_steps,
                                    n_updates_per_iteration=n_updates_per_iteration,
                                    gamma=gamma,
                                    gae_lambda=gae_lambda,
                                    vf_coef=vf_coef,
                                    bc_loss_coef=bc_loss_coef,
                                    bc_loss_coef_min=bc_loss_coef_min,
                                    bc_loss_coef_decay=bc_loss_coef_decay,
                                    entropy_coef=entropy_coef,
                                    entropy_coef_decay=entropy_coef_decay,
                                    clip=clip,
                                    actor_lr=actor_lr,
                                    critic_lr=critic_lr,
                                    )
    rl_agent.learn(n_iterations)