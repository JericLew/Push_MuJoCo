# Description: Implementing PPO-Clip based on OpenAI pseudocode
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

# Inputs to PPO Clip Algorithm: Network, Gym Environment, Hyperparameters
# Outputs: Stores training progress graphs 


import os
import time
import wandb
from PIL import Image

import torch
import numpy as np
from torch.optim import Adam


class PPOAgent():
    def __init__(self, venv, actor, critic, expert_actor, device=None):
        self.is_hyperparams_init = False
        
        ## Initialize Vectorized Environment
        self.venv = venv
        
        ## Initialize Actor and Critic Networks
        self.actor = actor
        self.critic = critic
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        ## Initialize Expert Actor
        self.expert_actor = expert_actor
        if self.expert_actor is not None:
            self.expert_actor.to(self.device)
            self.expert_actor.eval()  # Set expert actor to evaluation mode
            self.expert_actor.requires_grad_(False)

    def init_hyperparameters(self,
                             name="PPO-Push",
                             use_wandb=False,
                             save_model_interval=50,
                             save_image_interval=50,
                             n_steps=300,
                             n_envs=20,
                             batch_size=300,
                             grad_accumulation_steps=1,
                             n_updates_per_iteration=10,
                             gamma=0.999,
                             gae_lambda=0.95,
                             vf_coef=0.5,
                             bc_loss_coef=0.05,
                             entropy_coef=1e-2,
                             entropy_coef_decay=0.99,
                             clip=0.2,
                             actor_lr=1e-4,
                             critic_lr=5e-4,
                             ):
        self.name = name                                            # Name of the experiment
        self.use_wandb = use_wandb                                  # Use wandb for logging
        self.save_model_interval = save_model_interval              # Save model every n iterations
        self.save_image_interval = save_image_interval              # Save images every n iterations
        self.n_steps = n_steps                                      # number of steps in environment
        self.n_envs = n_envs                                        # number of parallel environments
        self.batch_size = batch_size                                # batch size for training
        self.grad_accumulation_steps = grad_accumulation_steps      # number of gradient accumulation steps
        self.n_updates_per_iteration = n_updates_per_iteration      # number of epochs per iteration
        self.gamma = gamma                                          # Discount Factor
        self.gae_lambda = gae_lambda                                # GAE Lambda
        self.vf_coef = vf_coef                                      # Value Function Coefficient
        self.bc_loss_coef = bc_loss_coef                            # Behavior Cloning Loss Coefficient
        self.entropy_coef = entropy_coef                            # Entropy Coefficient
        self.entropy_coef_decay = entropy_coef_decay                # Entropy Coefficient Decay
        self.clip = clip                                            # PPO clip parameter (recommended by paper)
        self.actor_lr = actor_lr                                    # Actor Learning Rate
        self.critic_lr = critic_lr                                  # Critic Learning Rate

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.is_hyperparams_init = True                             # Flag to check if hyperparameters are initialized

        ## Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="PPO-Push",
                config={
                    "save_model_interval": self.save_model_interval,
                    "save_image_interval": self.save_image_interval,
                    "n_steps": self.n_steps,
                    "n_envs": self.n_envs,
                    "batch_size": self.batch_size,
                    "grad_accumulation_steps": self.grad_accumulation_steps,
                    "n_updates_per_iteration": self.n_updates_per_iteration,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "vf_coef": self.vf_coef,
                    "bc_loss_coef": self.bc_loss_coef,
                    "entropy_coef": self.entropy_coef,
                    "entropy_coef_decay": self.entropy_coef_decay,
                    "clip": self.clip,
                    "actor_lr": self.actor_lr,
                    "critic_lr": self.critic_lr
                },
                name=name,
            )

    def learn(self, iterations):
        if self.is_hyperparams_init == False:
            assert False, "Hyperparameters not initialized. Please call init_hyperparameters() before learn()"
        for itr in range(iterations):
            start_time = time.time()
            print(f"Iteration {itr} Started")

            ## Collect Rollout Data
            rollout_obs, rollout_actions, rollout_log_probs, rollout_values, rollout_rewards, rollout_returns, rollout_advantages, rollout_firsts = self.rollout(itr)

            ## Summarize rewards
            avg_episode_reward, avg_best_reward, success_rate, num_episode_finished = self.summarize_rewards(rollout_firsts, rollout_rewards)
            print(f"Avg Episode Reward: {avg_episode_reward:.2f}, Avg Best Reward: {avg_best_reward:.2f}, Success Rate: {success_rate:.2f}, Num Episodes Finished: {num_episode_finished}")

            ## Save images every self.image_interval iterations (Only save env 0)
            if itr % self.save_image_interval == 0 and "image" in rollout_obs.keys():
                self.save_images(rollout_obs["image"][:, 0], itr) # Save images from env 0
                self.save_images(rollout_obs["image"][:, 1], itr+1) # Save images from env 1
                
            ## Convert to tensors (but do not move everything to GPU at once) and n_steps x n_envs x dim -> n_steps*n_envs x dim
            for key in rollout_obs.keys():
                rollout_obs[key] = torch.tensor(rollout_obs[key], dtype=torch.float32).flatten(0, 1)
            rollout_actions = torch.tensor(rollout_actions, dtype=torch.float32).flatten(0, 1)
            rollout_log_probs = torch.tensor(rollout_log_probs, dtype=torch.float32).flatten(0, 1)
            rollout_values = torch.tensor(rollout_values, dtype=torch.float32).flatten(0, 1)
            rollout_returns = torch.tensor(rollout_returns, dtype=torch.float32).flatten(0, 1)
            rollout_advantages = torch.tensor(rollout_advantages, dtype=torch.float32).flatten(0, 1)

            ## Define batch size
            num_samples = rollout_actions.size(0) # (n_steps*n_envs)
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size  # Calculate number of batches
            
            ## Loop to update the network
            average_bc_loss = 0
            average_actor_loss = 0
            average_critic_loss = 0
            average_entropy = 0
            average_total_loss = 0
            average_ratio = 0
            average_clipfrac = 0
            average_kl = 0
            for _ in range(self.n_updates_per_iteration):
                for batch_idx in range(num_batches):
                    ## Slice the data for the current batch
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, num_samples)

                    batch_obs = {key: value[start_idx:end_idx].to(self.device) for key, value in rollout_obs.items()}
                    batch_obs["image"] = batch_obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
                    batch_actions = rollout_actions[start_idx:end_idx].to(self.device)
                    batch_log_probs_old = rollout_log_probs[start_idx:end_idx].to(self.device)
                    # batch_vals_old = rollout_values[start_idx:end_idx].to(self.device) # Not necessary
                    batch_returns = rollout_returns[start_idx:end_idx].to(self.device)
                    batch_advantages = rollout_advantages[start_idx:end_idx].to(self.device)

                    ## Normalize batch advantages
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                    batch_advantages = torch.clamp(batch_advantages, -5, 5)

                    ## Get new log probabilities and values
                    batch_values_new, batch_log_probs_new, entropy = self.evaluate(batch_obs, batch_actions)

                    ## Calculate Ratios
                    ratios = torch.exp(batch_log_probs_new - batch_log_probs_old)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                    ## Get BC loss from privileged actor
                    bc_loss = 0
                    if self.expert_actor is not None:
                        dist = self.expert_actor.get_distribution(batch_obs) # (n_envs, action_dim)
                        imitation_action = dist.mean.detach() # (n_envs, action_dim)
                        _, bc_log_probs, _ = self.evaluate(batch_obs, imitation_action) # (n_envs, action_dim)
                        bc_loss = -bc_log_probs.mean()

                        if batch_idx == 0:
                            print(f"BC Loss: {bc_loss.item()}, Imitation Action: {imitation_action[0]}, bc_log_probs: {bc_log_probs[0]}, Action: {batch_actions[0]}, Log Probs: {batch_log_probs_new[0]}")

                    ## Loss Function
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = torch.nn.MSELoss()(batch_values_new, batch_returns)
                    total_loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy.mean() + self.bc_loss_coef * bc_loss

                    ## Logging
                    with torch.no_grad():
                        average_bc_loss += bc_loss.item() if self.expert_actor is not None else 0
                        average_actor_loss += actor_loss.item()
                        average_critic_loss += critic_loss.item()
                        average_entropy += entropy.mean().item()
                        average_total_loss += total_loss.item()
                        average_ratio += ratios.mean().item()
                        average_clipfrac += ((ratios - 1.0).abs() > self.clip).float().mean().item()
                        average_kl += ((ratios - 1) - (batch_log_probs_new - batch_log_probs_old)).mean()

                    ## Update networks
                    total_loss.backward()
                    if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                        self.actor_optim.step()
                        self.critic_optim.step()
                        self.actor_optim.zero_grad()
                        self.critic_optim.zero_grad()
                        print(f"Batch {batch_idx} gradient step completed")
            
            ## Save model every self.save_model_interval iterations
            if itr % self.save_model_interval == 0:
                self.save_model(f"weights/{itr}/")
                print(f"Model saved at iteration {itr}")

            ## Logging Average losses
            average_bc_loss /= num_batches * self.n_updates_per_iteration
            average_actor_loss /= num_batches * self.n_updates_per_iteration
            average_critic_loss /= num_batches * self.n_updates_per_iteration
            average_entropy /= num_batches * self.n_updates_per_iteration
            average_total_loss /= num_batches * self.n_updates_per_iteration
            print(f"BC Loss: {average_bc_loss:.7f}, Actor Loss: {average_actor_loss:.7}, Critic Loss: {average_critic_loss:.7f}, Entropy: {average_entropy:.7f}, Total Loss: {average_total_loss:.7f}")
            
            ## Logging PPO stats
            average_ratio /= num_batches * self.n_updates_per_iteration
            average_clipfrac /= num_batches * self.n_updates_per_iteration
            average_kl /= num_batches * self.n_updates_per_iteration
            y_pred = batch_values_new.detach().cpu().numpy()
            y_true = batch_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_variance = 1 - (np.var(y_true - y_pred) / var_y) if var_y > 0 else np.nan
            print(f"Average Ratio: {average_ratio:.7f}, Average Clipfrac: {average_clipfrac:.7f}, Average KL: {average_kl:.7f}, Explained Variance: {explained_variance:.7f}")

            ## Logging Decaying Terms
            print(f"Entropy Coefficient: {self.entropy_coef}")

            ## Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                    "bc_loss": average_bc_loss,
                    "actor_loss": average_actor_loss,
                    "critic_loss": average_critic_loss,
                    "entropy": average_entropy,
                    "total_loss": average_total_loss,
                    "ratio": average_ratio,
                    "clipfrac": average_clipfrac,
                    "kl": average_kl,
                    "explained_variance": explained_variance,
                    "avg_episode_reward": avg_episode_reward,
                    "avg_best_reward": avg_best_reward,
                    "success_rate": success_rate,
                    "num_episode_finished": num_episode_finished,
                    "entropy_coef": self.entropy_coef,
                    },
                    step=itr,
                    commit=True,
                )

            ## Decay Terms
            self.entropy_coef *= self.entropy_coef_decay

            ## Log training time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Iteration {itr} completed in {elapsed_time:.2f} seconds")
            print(f"==========================")

    def rollout(self, itr):
        ## Init rollout holders
        rollout_obs = {} # (n_steps, n_envs, obs_dim) - stores observations at each step
        rollout_actions = np.zeros((self.n_steps, self.n_envs, self.venv.action_space.shape[1])) # (n_steps, n_envs, act_dim) - stores actions taken at each step
        rollout_log_probs = np.zeros((self.n_steps, self.n_envs)) # (n_steps, n_envs) - stores log probabilities of actions taken at each step
        rollout_values = np.zeros((self.n_steps, self.n_envs)) # (n_steps, n_envs) - stores value function estimates at each step
        rollout_rewards = np.zeros((self.n_steps, self.n_envs)) # (n_steps, n_envs) - stores rewards at each step
        rollout_terminated = np.zeros((self.n_steps, self.n_envs)) # (n_steps, n_envs) - stores termination flags at each step
        rollout_returns = np.zeros((self.n_steps, self.n_envs)) # (n_steps, n_envs) - stores returns at each step
        rollout_advantages = np.zeros((self.n_steps, self.n_envs)) # (n_steps, n_envs) - stores advantages at each step
        rollout_firsts = np.zeros((self.n_steps + 1, self.n_envs)) # (n_steps, n_envs) - stores first step flags at each step

        ## Reset environment
        options = None
        prev_obs, prev_info = self.venv.reset(options=options)
        rollout_firsts[0] = 1 # First step

        ## Data collection loop
        for step in range(self.n_steps):
            ## Collect observation
            for key in prev_obs.keys():
                if key not in rollout_obs: # Initialize batch_obs for each key if empty
                    if key == "image": # handle image separately
                        rollout_obs[key] = np.zeros((self.n_steps, self.n_envs, *prev_obs[key].shape[1:])) # (shape[1:] = image_dim) N x H x W x C
                    else:
                        rollout_obs[key] = np.zeros((self.n_steps, self.n_envs, prev_obs[key].shape[1])) # (shape[1] = key_dim)
                rollout_obs[key][step] = prev_obs[key]

            ## Step the environment
            action, log_prob, value = self.get_action(prev_obs) # get action from actor network
            obs, rew, terminated, truncated, info = self.venv.step(action)

            ## Collect data
            rollout_actions[step] = action
            rollout_log_probs[step] = log_prob
            rollout_values[step] = value
            rollout_rewards[step] = rew
            rollout_terminated[step] = terminated
            rollout_firsts[step + 1] = terminated | truncated # next step is first if terminated or truncated

            ## Update previous observation
            prev_obs = obs
            prev_info = info
        
        ## Compute returns and advantages
        _, _, last_value = self.get_action(prev_obs) # Get value for last observation (S_{T+1})
        rollout_returns, rollout_advantages = self.compute_gae(rollout_rewards, rollout_values, rollout_terminated, last_value)
        
        return rollout_obs, rollout_actions, rollout_log_probs, rollout_values, rollout_rewards, rollout_returns, rollout_advantages, rollout_firsts
    
    def compute_gae(self, rewards, values, terminated, last_value):
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(rewards.shape[0])): # iterate over time steps (n_steps)
            if t == rewards.shape[0] - 1: # last step
                nextvalues = last_value
            else:
                nextvalues = values[t + 1]
            nonterminal = 1.0 - terminated[t]
            ## delta_t = r_t + gamma*V(S_{t+1}) - V(S_t) NOTE: V(S_{t+1{}) = 0 if terminal
            delta = rewards[t] + self.gamma * nextvalues * nonterminal - values[t]
            ## A_t = delta_t + gamma*lambda*delta_{t+1} + gamma^2*lambda^2*delta_{t+2} + ...
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    def get_action(self, obs):
        obs = {key: torch.from_numpy(obs[key].astype(np.float32)).to("cuda").float() for key in obs.keys()}  # Convert to torch tensors
        obs["image"] = obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
        dist = self.actor.get_distribution(obs)  # Get distribution from actor network
        value = self.critic(obs)  # Get value from critic network
        action = dist.sample()  # shape: (batch_size, action_dim)
        # action = torch.clamp(action, self.actor.action_low, self.actor.action_high)  # Clip action to action space
        log_prob = dist.log_prob(action)  # shape: (batch_size,)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().squeeze(1).numpy()
    
    def evaluate(self, batch_obs, batch_actions):
        dist = self.actor.get_distribution(batch_obs)  # Get distribution from actor network
        value = self.critic(batch_obs)
        log_prob = dist.log_prob(batch_actions)  # shape: (batch_size,)
        entropy = dist.entropy()
        return value.squeeze(1), log_prob, entropy

    def summarize_rewards(self, rollout_firsts, rollout_rewards):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(rollout_firsts[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start >= 1: # NOTE might be >= 1 instead of > 1
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            rollout_rewards_split = [
                rollout_rewards[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(rollout_rewards_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in rollout_rewards_split]
            )
            episode_best_reward = np.array(
                [
                    np.max(rewards) for rewards in rollout_rewards_split
                ]
            )
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= 0.9 # NOTE Success reward == 1.0 - some small negative reward
            )
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            print(f"Warning: No episode finished in this iteration!")
        return avg_episode_reward, avg_best_reward, success_rate, num_episode_finished

    def save_images(self, env_images, itr):
        """
        Save images from rollout_obs into a structured directory.

        Args:
            env_images (numpy.ndarray): Images to be saved. (n_steps, num_images, H, W, C)
            itr (int): Current iteration number.
            image_interval (int): Interval at which images should be saved.
        """
        # Define the base directory for saving images
        base_dir = f"images/itr_{itr}"
        os.makedirs(base_dir, exist_ok=True)

        rollout_images_shape = env_images.shape
        number_of_images = rollout_images_shape[1]  # Number of images in the batch
        for i in range(number_of_images):
            # Create a directory for each image
            image_dir = os.path.join(base_dir, f"image_{i}")
            os.makedirs(image_dir, exist_ok=True)

            # Iterate through the steps and save each image
            for step in range(rollout_images_shape[0]):
                img_pil = Image.fromarray((env_images[step, i] * 255).astype("uint8"))
                img_pil.save(os.path.join(image_dir, f"step_{step}.png"))

    def save_model(self, folderpath):
        """Save the model weights to the specified filepath."""
        # Ensure the directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folderpath = os.path.join(script_dir, folderpath)
        os.makedirs(os.path.dirname(folderpath), exist_ok=True)
        actor_path = os.path.join(folderpath, "actor.pth")
        critic_path = os.path.join(folderpath, "critic.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Model saved to {folderpath}")


if __name__ == "__main__":
    pass