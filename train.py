import os
import torch
import numpy as np
import gymnasium as gym

from PPO import PPOAgent
from model.common.cnn import ImageEncoder, DualImageEncoder
from model.push_actor import PushNNActor, PushNNPrivilegedActor
from model.push_critic import PushNNCritic, PushNNPrivilegedCritic

if __name__ == "__main__":
    ## PPO Hyperparameters
    use_wandb = True
    max_episode_steps = 300
    n_envs = 20 #12
    vectorization_mode = "async"
    save_model_interval = 50
    save_image_interval = 50
    batch_size = 300
    n_iterations = 10000

    # name = "privileged-delta_action-scaled_tanh"
    # n_updates_per_iteration = 10
    # gamma = 0.999
    # gae_lambda = 0.95
    # imitation_reward_coef = 0.0
    # entropy_coef = 1e-2
    # entropy_coef_decay = 0.99
    # clip = 0.2
    # actor_lr = 1e-4
    # critic_lr = 5e-4

    name = "imitation-delta_action-scaled_tanh"
    n_updates_per_iteration = 10
    gamma = 0.999
    gae_lambda = 0.95
    imitation_reward_coef = 0.005
    entropy_coef = 1e-2
    entropy_coef_decay = 0.99
    clip = 0.2
    actor_lr = 5e-5
    critic_lr = 1e-4

    ## Environment Hyperparameters
    privileged = False # Training with privileged information
    random_object_pos = False

    ## Make Environment
    gym.register(
        id="PickPlaceCustomEnv-v0",
        entry_point="push_custom:PickPlaceCustomEnv",
        max_episode_steps=max_episode_steps,
    )

    xml_path = "franka_emika_panda/scene_push.xml"
    venv = gym.make_vec("PickPlaceCustomEnv-v0",
                        xml_path=xml_path,
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
    print(f"State Space: {venv.observation_space['state']}")
    print(f"Image Space: {venv.observation_space['image']}")
    print(f"Privileged Space: {venv.observation_space['privileged']}")
    print(f"Action Space: {venv.action_space}")

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

    ## Handle Privileged or Non-Privileged Training, if not privileged use imitation learning
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
        critic = PushNNPrivilegedCritic(
            privileged_dim=privileged_dim,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
            dropout=0.0,
        )
        expert_actor = None # No imitation learning
    else:
        ## Network Hyperparameters
        fixed_std = 0.06
        learn_fixed_std = True
        std_min = 0.005
        std_max = 0.1

        image_encoder_actor = DualImageEncoder(
            image_input_shape=image_dim,
            feature_dim=256,
        )
        image_encoder_critic = DualImageEncoder(
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
        critic = PushNNCritic(
            backbone=image_encoder_critic,
            state_dim=state_dim,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
            visual_feature_dim=128,
            dropout=0.0,
        )
        expert_actor = PushNNPrivilegedActor(
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
        pth_path = os.path.expanduser("~/Jeric/Push_MuJoCo/saved_pth/privileged_actor.pth")
        expert_actor.load_state_dict(torch.load(pth_path))
        expert_actor.to("cpu")
        expert_actor.eval()
    
    ## Remake environment to possibly add expert actor for imitation learning
    ## TODO: kinda hacky, need to find a better way to do this
    venv.close()
    vector_kwargs = {
        "context": "forkserver", # spawn, fork, forkserver
    }
    venv = gym.make_vec("PickPlaceCustomEnv-v0",
                        xml_path=xml_path,
                        expert_actor=expert_actor,
                        privileged=privileged,
                        random_object_pos=random_object_pos,
                        render_mode=None,
                        max_episode_steps=max_episode_steps,
                        num_envs=n_envs,
                        vectorization_mode=vectorization_mode,
                        vector_kwargs=vector_kwargs,
                        )
    
    ## Initialize PPO Agent
    rl_agent = PPOAgent(venv, actor=actor, critic=critic)
    rl_agent.init_hyperparameters(use_wandb=use_wandb,
                                    name=name,
                                    save_model_interval=save_model_interval,
                                    save_image_interval=save_image_interval,
                                    n_steps=max_episode_steps,
                                    n_envs=n_envs,
                                    batch_size=batch_size,
                                    n_updates_per_iteration=n_updates_per_iteration,
                                    gamma=gamma,
                                    gae_lambda=gae_lambda,
                                    imitation_reward_coef=imitation_reward_coef,
                                    entropy_coef=entropy_coef,
                                    entropy_coef_decay=entropy_coef_decay,
                                    clip=clip,
                                    actor_lr=actor_lr,
                                    critic_lr=critic_lr,
                                    privileged=privileged,
                                    random_object_pos=random_object_pos,
                                    )
    rl_agent.learn(n_iterations)