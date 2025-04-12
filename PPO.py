# Description: Implementing PPO-Clip based on OpenAI pseudocode
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

# Inputs to PPO Clip Algorithm: Network, Gym Environment, Hyperparameters
# Outputs: Stores training progress graphs 

'''
Steps:
1. Collect Data
2. Summarize Rewards
3. Get Value from Critic Network
4. Get log probabilities of actions taken
5. Calculate Advantage
6. loop for n epochs:
    7. Get Value from Critic Network
    8. Calculate Action Probability Ratio
    9. Calculate Surrogate Loss
    10. Calculate Clipped Surrogate Loss
    11. Calculate Actor Loss
    12. Calculate Critic Loss
    13. Perform Backpropagation for Actor and Critic Networks
14. Update Actor and Critic Networks
'''
import os
import time
import wandb
from PIL import Image

import torch
import numpy as np
from torch.optim import Adam
from torch.distributions import Normal, Independent

from network import FeedForwardNN # PPO Network
from push_nn import PushNN

class PPOAgent():
    def __init__(self, venv, actor, critic, device=None):
        self.is_hyperparams_init = False
        
        # Initialize Vectorized Environment
        self.venv = venv
        
        # Initialize Actor and Critic Networks
        self.actor = actor
        self.critic = critic
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def init_hyperparameters(self,
                             name="PPO-Push",
                             save_model_interval=100,
                             save_image_interval=20,
                             n_steps=500,
                             n_envs=12,
                             batch_size=128,
                             n_updates_per_iteration=5,
                             gamma=0.95,
                             gae_lambda=0.95,
                             entropy_coef=0.0001,
                             clip=0.2,
                             actor_lr=0.0005,
                             critic_lr=0.001):
        self.save_model_interval = save_model_interval              # Save model every n iterations
        self.save_image_interval = save_image_interval              # Save images every n iterations
        self.n_steps = n_steps                                      # number of steps in environment
        self.n_envs = n_envs                                        # number of parallel environments
        self.batch_size = batch_size                                # batch size for training
        self.n_updates_per_iteration = n_updates_per_iteration      # number of epochs per iteration
        self.gamma = gamma                                          # Discount Factor
        self.gae_lambda = gae_lambda                                # GAE Lambda
        self.entropy_coef = entropy_coef                            # Entropy Coefficient
        self.clip = clip                                            # PPO clip parameter (recommended by paper)
        self.actor_lr = actor_lr                                    # Actor Learning Rate
        self.critic_lr = critic_lr                                  # Critic Learning Rate

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        # Init action space
        # Mean is in range [-1, 1] due to Tanh activation, scale to [action_low, action_high]
        ## THIS IS FOR DELTA ACTION can easily do it for absolute action
        self.action_high = 3 / 180 * np.pi # 3 degrees in radians / 10*2(10^-3) seconds = 150 degrees/s
        self.action_low = -3 / 180 * np.pi # 3 degrees in radians / 10*2(10^-3) seconds = 150 degrees/s
        self.action_range = self.action_high - self.action_low

        self.is_hyperparams_init = True                             # Flag to check if hyperparameters are initialized

        wandb.init(
            project="PPO-Push",
            config={
                "save_model_interval": self.save_model_interval,
                "save_image_interval": self.save_image_interval,
                "n_steps": self.n_steps,
                "n_envs": self.n_envs,
                "batch_size": self.batch_size,
                "n_updates_per_iteration": self.n_updates_per_iteration,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "entropy_coef": self.entropy_coef,
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
                rollout_obs[key] = torch.tensor(rollout_obs[key], dtype=torch.float).flatten(0, 1)
            rollout_actions = torch.tensor(rollout_actions, dtype=torch.float).flatten(0, 1)
            rollout_log_probs = torch.tensor(rollout_log_probs, dtype=torch.float).flatten(0, 1)
            rollout_values = torch.tensor(rollout_values, dtype=torch.float).flatten(0, 1)
            rollout_returns = torch.tensor(rollout_returns, dtype=torch.float).flatten(0, 1)
            rollout_advantages = torch.tensor(rollout_advantages, dtype=torch.float).flatten(0, 1)

            ## Normalize advantages
            rollout_advantages = (rollout_advantages - rollout_advantages.mean()) / (rollout_advantages.std() + 1e-10)
            # rollout_advantages = torch.clamp(rollout_advantages, -5, 5) # Clip advantages to avoid exploding gradients

            ## Define batch size
            num_samples = rollout_actions.size(0) # (n_steps*n_envs)
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size  # Calculate number of batches
            
            ## Loop to update the network
            average_actor_loss = 0
            average_critic_loss = 0
            average_entropy = 0
            average_total_loss = 0
            average_ratio = 0
            average_clipfrac = 0
            average_kl = 0
            for _ in range(self.n_updates_per_iteration):
                for batch_idx in range(num_batches):
                    # Slice the data for the current batch
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, num_samples)

                    batch_obs = {key: value[start_idx:end_idx].to(self.device) for key, value in rollout_obs.items()}
                    batch_obs["image"] = batch_obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
                    batch_actions = rollout_actions[start_idx:end_idx].to(self.device)
                    batch_log_probs_old = rollout_log_probs[start_idx:end_idx].to(self.device)
                    # batch_vals_old = rollout_values[start_idx:end_idx].to(self.device) # Not necessary
                    batch_returns = rollout_returns[start_idx:end_idx].to(self.device)
                    batch_advantages = rollout_advantages[start_idx:end_idx].to(self.device)

                    # Perform forward pass and calculate losses
                    batch_values_new, batch_log_probs_new, entropy = self.evaluate(batch_obs, batch_actions)

                    ratios = torch.exp(batch_log_probs_new - batch_log_probs_old)

                    # Unclipped Surrogate Loss
                    surr1 = ratios * batch_advantages

                    # Clipped Surrogate Loss
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                    # Loss Function
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = torch.nn.MSELoss()(batch_values_new, batch_returns)
                    total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                    # Logging
                    average_actor_loss += actor_loss.item()
                    average_critic_loss += critic_loss.item()
                    average_entropy += entropy.item()
                    average_total_loss += total_loss.item()
                    average_ratio += ratios.mean().item()
                    average_clipfrac += ((ratios - 1.0).abs() > self.clip).float().mean().item()
                    average_kl += ((ratios - 1.0).abs()).mean().item()

                    # Update the network
                    self.actor_optim.zero_grad()
                    self.critic_optim.zero_grad()
                    total_loss.backward()
                    self.actor_optim.step()
                    self.critic_optim.step()
            
            ## Save model every self.save_model_interval iterations
            if itr % self.save_model_interval == 0:
                self.save_model(f"weights/{itr}/")
                print(f"Model saved at iteration {itr}")

            ## Logging Average losses
            average_actor_loss /= num_batches * self.n_updates_per_iteration
            average_critic_loss /= num_batches * self.n_updates_per_iteration
            average_entropy /= num_batches * self.n_updates_per_iteration
            average_total_loss /= num_batches * self.n_updates_per_iteration
            print(f"Average Actor Loss: {average_actor_loss:.7}, Average Critic Loss: {average_critic_loss:.7f}, Average Entropy: {average_entropy:.7f}, Average Total Loss: {average_total_loss:.7f}")
            
            ## Logging PPO stats
            average_ratio /= num_batches * self.n_updates_per_iteration
            average_clipfrac /= num_batches * self.n_updates_per_iteration
            average_kl /= num_batches * self.n_updates_per_iteration
            y_pred = batch_values_new.detach().cpu().numpy()
            y_true = batch_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_variance = 1 - (np.var(y_true - y_pred) / var_y) if var_y > 0 else np.nan
            print(f"Average Ratio: {average_ratio:.7f}, Average Clipfrac: {average_clipfrac:.7f}, Average KL: {average_kl:.7f}, Explained Variance: {explained_variance:.7f}")

            ## Log to wandb
            wandb.log(
                {
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
                },
                step=itr,
                commit=True,
            )

            ## Log training time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Iteration {itr} completed in {elapsed_time:.2f} seconds")

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
        level = 0
        if itr > 100:
            level = 1
        if itr > 200:
            level = 2
        if itr > 300:
            level = 3
        options = {"level": level}
        print(f"Level: {level}")
        obs, _ = self.venv.reset(options=options)
        rollout_firsts[0] = 1 # First step

        ## Data collection loop
        for step in range(self.n_steps):
            ## Collect observation
            for key in obs.keys():
                if key not in rollout_obs: # Initialize batch_obs for each key if empty
                    if key == "state":
                        rollout_obs[key] = np.zeros((self.n_steps, self.n_envs, obs[key].shape[1])) # (shape[1] = state_dim)
                    elif key == "image":
                        rollout_obs[key] = np.zeros((self.n_steps, self.n_envs, *obs[key].shape[1:])) # (shape[1:] = image_dim) N x H x W x C
                rollout_obs[key][step] = obs[key]

            ## Step the environment
            action, log_prob, value = self.get_action(obs) # get action from actor network
            obs, rew, terminated, truncated, _ = self.venv.step(action)

            ## Collect data
            rollout_actions[step] = action
            rollout_log_probs[step] = log_prob
            rollout_values[step] = value
            rollout_rewards[step] = rew
            rollout_terminated[step] = terminated
            rollout_firsts[step + 1] = terminated | truncated # next step is first if terminated or truncated
        
        ## Compute returns and advantages
        _, _, last_value = self.get_action(obs) # Get value for last observation (S_{T+1})
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
            # delta_t = r_t + gamma*V(S_{t+1}) - V(S_t) NOTE: V(S_{t+1{}) = 0 if terminal
            delta = rewards[t] + self.gamma * nextvalues * nonterminal - values[t]
            # A_t = delta_t + gamma*lambda*delta_{t+1} + gamma^2*lambda^2*delta_{t+2} + ...
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    # Joint speed limits in degrees/s A1: 150, A2: 150, A3: 150, A4: 150, A5: 180, A6: 180, A7: 180
    def get_action(self, obs):
        obs = {key: torch.from_numpy(obs[key]).to("cuda").float() for key in obs.keys()}  # Convert to torch tensors
        obs["image"] = obs["image"].permute(0, 1, 4, 2, 3)  # Change from (B, N, H, W, C) to (B, N, C, H, W)
        (mean, std), _ = self.actor(obs)  # Get mean and std from actor network
        (_, _), value = self.critic(obs)  # Get value from critic network
        scaled_mean = 0.5 * (mean + 1) * self.action_range + self.action_low # NOTE assuming center of action space is 0 and action space is symmetric
        scaled_std = 0.5 * std * self.action_range
        dist = Independent(Normal(scaled_mean, scaled_std), 1)  # 1 = number of reinterpreted batch dims
        action = dist.sample()  # shape: (batch_size, 7)
        log_prob = dist.log_prob(action)  # shape: (batch_size,)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().squeeze(1).numpy()
    
    def evaluate(self, batch_obs, batch_actions):
        (mean, std), _ = self.actor(batch_obs)
        (_, _), value = self.critic(batch_obs)
        scaled_mean = 0.5 * (mean + 1) * self.action_range + self.action_low # NOTE assuming center of action space is 0 and action space is symmetric
        scaled_std = 0.5 * std * self.action_range
        dist = Independent(Normal(scaled_mean, scaled_std), 1)  # 1 = number of reinterpreted batch dims
        log_prob = dist.log_prob(batch_actions)  # shape: (batch_size,)
        entropy = dist.entropy().mean()  # Entropy averaged over batch
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
    # Example usage of the PPOAgent class
    # Initialize the environment and agent
    import gymnasium as gym

    ## Hyperparameters
    name = "PPO-Push-delta_action-scaled_tanh"
    max_episode_steps = 300 #500 # 1000
    n_envs = 20 #12
    vectorization_mode = "async"
    save_model_interval = 50
    save_image_interval = 20
    batch_size = 300
    n_updates_per_iteration = 3
    gamma = 0.99
    gae_lambda = 0.98
    entropy_coef = 1e-5
    clip = 0.2
    actor_lr = 0.0001
    critic_lr = 0.0005
    n_iterations = 10000

    gym.register(
        id="PickPlaceCustomEnv-v0",
        entry_point="push_custom:PickPlaceCustomEnv",
        max_episode_steps=max_episode_steps,
    )

    xml_path = "franka_emika_panda/scene_push.xml"
    venv = gym.make_vec("PickPlaceCustomEnv-v0",
                        xml_path=xml_path,
                        render_mode=None,
                        max_episode_steps=max_episode_steps,
                        num_envs=n_envs,
                        vectorization_mode=vectorization_mode,
                        )

    state_dim = venv.observation_space["state"].shape[1]
    image_dim = venv.observation_space["image"].shape[2:] # (B, N, H, W, C) -> (H, W, C)
    image_dim = (image_dim[2], image_dim[0], image_dim[1]) # (H, W, C) -> (C, H, W)
    action_dim = venv.action_space.shape[1]
    print("State Space: ", state_dim)
    print("Image Space: ", image_dim)
    print("Action Space: ", action_dim)
    actor = PushNN(state_dim=state_dim, image_dim=image_dim, action_dim=action_dim)
    critic = PushNN(state_dim=state_dim, image_dim=image_dim, action_dim=action_dim)

    rl_agent = PPOAgent(venv, actor=actor, critic=critic)
    rl_agent.init_hyperparameters(name=name,
                                  save_model_interval=save_model_interval,
                                  save_image_interval=save_image_interval,
                                  n_steps=max_episode_steps,
                                  n_envs=n_envs,
                                  batch_size=batch_size,
                                  n_updates_per_iteration=n_updates_per_iteration,
                                  gamma=gamma,
                                  gae_lambda=gae_lambda,
                                  entropy_coef=entropy_coef,
                                  clip=clip,
                                  actor_lr=actor_lr,
                                  critic_lr=critic_lr)
    rl_agent.learn(n_iterations)

