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

import torch
import numpy as np
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.distributions import Normal, Independent

from network import FeedForwardNN # PPO Network
from push_nn import PushNN

class PPOAgent():
    def __init__(self, venv, network):
        # Initialize Hyperparameters
        self._init_hyperparameters()

        self.logger = {}
        
        # Initialize Vectorized Environment
        self.venv = venv
        
        # Initialize Actor and Critic Networks
        self.network = network
        self.device = self.network.device

        self.network_optim = Adam(self.network.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.n_steps = 500                      # number of steps in environment
        self.n_envs = 50                        # number of parallel environments
        self.batch_size = 128                   # batch size for training
        self.n_updates_per_iteration = 5        # number of epochs per iteration
        self.gamma = 0.95                       # Discount Factor
        self.gae_lambda = 0.95                  # GAE Lambda
        self.entropy_coef = 0.01                # Entropy Coefficient
        self.clip = 0.2                         # PPO clip parameter (recommended by paper)
        self.lr = 0.0001                       # Learning Rate

    def learn(self, iterations):
        for itr in range(iterations):
            print(f"Iteration: {itr}")

            ## Collect Rollout Data
            rollout_obs, rollout_actions, rollout_log_probs, rollout_values, rollout_returns, rollout_advantages = self.rollout()

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

            ## Define batch size
            batch_size = self.batch_size
            num_samples = rollout_actions.size(0) # (n_steps*n_envs)
            num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches
            
            ## Loop to update the network
            for _ in range(self.n_updates_per_iteration):
                for batch_idx in range(num_batches):
                    # Slice the data for the current batch
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)

                    batch_obs = {key: value[start_idx:end_idx].to(self.device) for key, value in rollout_obs.items()}
                    batch_actions = rollout_actions[start_idx:end_idx].to(self.device)
                    batch_log_probs_old = rollout_log_probs[start_idx:end_idx].to(self.device)
                    # batch_vals_old = rollout_values[start_idx:end_idx].to(self.device) # Not necessary
                    batch_returns = rollout_returns[start_idx:end_idx].to(self.device)
                    batch_advantages = rollout_advantages[start_idx:end_idx].to(self.device)

                    # Perform forward pass and calculate losses
                    batch_values_new, batch_log_probs_new = self.evaluate(batch_obs, batch_actions)

                    ratios = torch.exp(batch_log_probs_new - batch_log_probs_old)

                    # Unclipped Surrogate Loss
                    surr1 = ratios * batch_advantages

                    # Clipped Surrogate Loss
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                    # Loss Function
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = torch.nn.MSELoss()(batch_values_new, batch_returns)
                    total_loss = actor_loss + 0.5 * critic_loss

                    print(f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Total Loss: {total_loss.item()}")

                    # Update the network
                    self.network_optim.zero_grad()
                    total_loss.backward()
                    self.network_optim.step()

    def rollout(self):
        ## Init rollout holders
        rollout_obs = {} # (n_steps, n_envs, obs_dim) - stores observations at each step
        rollout_actions = np.zeros((self.n_steps, self.venv.num_envs, self.venv.action_space.shape[1])) # (n_steps, n_envs, act_dim) - stores actions taken at each step
        rollout_log_probs = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores log probabilities of actions taken at each step
        rollout_values = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores value function estimates at each step
        rollout_rewards = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores rewards at each step
        rollout_terminated = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores termination flags at each step
        rollout_dones = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores done flags at each step
        rollout_returns = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores returns at each step
        rollout_advantages = np.zeros((self.n_steps, self.venv.num_envs)) # (n_steps, n_envs) - stores advantages at each step

        ## Reset environment
        obs, _ = self.venv.reset()

        ## Data collection loop
        for step in range(self.n_steps):
            ## Collect observation
            for key in obs.keys():
                if key not in rollout_obs: # Initialize batch_obs for each key if empty
                    if key == "state":
                        rollout_obs[key] = np.zeros((self.n_steps, self.venv.num_envs, obs[key].shape[1])) # (shape[1] = state_dim)
                    elif key == "image":
                        rollout_obs[key] = np.zeros((self.n_steps, self.venv.num_envs, *obs[key].shape[1:])) # (shape[1:] = image_dim) H x W x C
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
            rollout_dones[step] = terminated | truncated

        print("batch rewards: ", rollout_rewards.max()) # Simple check at performance

        ## Compute returns and advantages
        _, _, last_value = self.get_action(obs) # Get value for last observation (S_{T+1})
        rollout_returns, rollout_advantages = self.compute_gae(rollout_rewards, rollout_values, rollout_terminated, last_value)
        
        return rollout_obs, rollout_actions, rollout_log_probs, rollout_values, rollout_returns, rollout_advantages
    
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

    def get_action(self, obs):
        obs = {key: torch.from_numpy(obs[key]).to("cuda").float() for key in obs.keys()}  # Convert to torch tensors
        (mean, std), value = self.network(obs)
        dist = Independent(Normal(mean, std), 1)  # 1 = number of reinterpreted batch dims
        action = dist.sample()  # shape: (batch_size, 7)
        log_prob = dist.log_prob(action)  # shape: (batch_size,)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().squeeze(1).numpy()
    
    def evaluate(self, batch_obs, batch_actions):
        (mean, std), value = self.network(batch_obs)
        dist = Independent(Normal(mean, std), 1)  # 1 = number of reinterpreted batch dims
        log_prob = dist.log_prob(batch_actions)  # shape: (batch_size,)
        return value, log_prob


if __name__ == "__main__":
    # Example usage of the PPOAgent class
    # Initialize the environment and agent

    import gymnasium as gym

    gym.register(
        id="PickPlaceCustomEnv-v0",
        entry_point="push_custom:PickPlaceCustomEnv",
        max_episode_steps=1000,
    )

    xml_path = "franka_emika_panda/scene_push.xml"
    venv = gym.make_vec("PickPlaceCustomEnv-v0",
                        xml_path=xml_path,
                        render_mode="camera",
                        max_episode_steps=500,
                        num_envs=20,
                        vectorization_mode="sync",
                        )


    state_dim = venv.observation_space["state"].shape[1]
    image_dim = venv.observation_space["image"].shape[1:]
    action_dim = venv.action_space.shape[1]
    print("State Space: ", state_dim)
    print("Image Space: ", image_dim)
    print("Action Space: ", action_dim)
    network = PushNN(state_dim=state_dim, image_dim=image_dim, action_dim=action_dim)

    rl_agent = PPOAgent(venv, network)
    rl_agent.learn(10000)

