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
        self.timesteps_per_batch = 10000        # timesteps per batch
        self.max_timesteps_per_episode = 1000   # timesteps per episode
        self.gamma = 0.95                       # Discount Factor
        self.n_updates_per_iteration = 5        # number of epochs per iteration
        self.clip = 0.2                         # PPO clip parameter (recommended by paper)
        self.lr = 0.00001                         # Learning Rate

    def learn(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            # Collect Data, Summarize Rewards, Get Value from Critic Network, Get log probabilities of actions taken
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_vals = self.rollout()
            print(f"Returns: {batch_rtgs}")

            # Calculate advantage
            # If A_k is positive, action was better than expected; If A_k negative, action was worse than expected
            A_k = batch_rtgs - batch_vals

            # Normalize advantages
            # Advantage values can have high variance, normalization helps stabilize training
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # The loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):

                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Unclipped Surrogate Loss
                # Policy Probability Ratio * Advantage Estimate
                surr1 = ratios * A_k

                # Clipped Surrogate Loss
                # Torch clamp: limits (or clamps) the values of a tensor within a specified range 
                # torch.clamp(input, min, max)
                surr2 = torch.clamp(ratios, 1-self.clip, 1 + self.clip) * A_k

                # Loss Function
                # Clipping by taking the min of the two surrogate losses 
                actor_loss = (-torch.min(surr1, surr2)).mean()

                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                total_loss = actor_loss + 0.5 * critic_loss

                # Update the network
                self.network_optim.zero_grad()
                total_loss.backward()
                self.network_optim.step()

    def rollout(self):
        batch_obs = {} # (n_steps, n_envs, obs_dim) - stores observations at each step
        batch_acts = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs, self.venv.action_space.shape[1])) # (n_steps, n_envs, act_dim) - stores actions taken at each step
        batch_log_probs = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs)) # (n_steps, n_envs) - stores log probabilities of actions taken at each step
        batch_rews = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs)) # (n_steps, n_envs) - stores rewards at each step
        batch_rtgs = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs)) # (n_steps, n_envs) - stores rewards-to-go at each step
        batch_vals = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs)) # (n_steps, n_envs) - stores value function estimates at each step
        obs, _ = self.venv.reset() # reset environment and get initial observation
        
        for step in range(self.max_timesteps_per_episode):
            ## Collect observation
            for key in obs.keys():
                if key not in batch_obs: # Initialize batch_obs for each key if empty
                    if key == "state":
                        batch_obs[key] = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs, obs[key].shape[1]))
                    elif key == "image":
                        batch_obs[key] = np.zeros((self.max_timesteps_per_episode, self.venv.num_envs, *obs[key].shape[1:]))
                batch_obs[key][step] = obs[key] # Store observation at each step

            ## Step the environment
            action, log_prob, value = self.get_action(obs) # get action from actor network
            obs, rew, terminated, truncated, _ = self.venv.step(action)

            # Store action, log probability of action taken, and reward at each step
            done = terminated | truncated
            batch_acts[step] = action
            batch_log_probs[step] = log_prob # store log probability of action taken
            batch_rews[step] = rew # store reward at each step
            batch_vals[step] = value

        batch_rtgs = self.compute_rtgs(batch_rews) # compute rewards-to-go

        # Convert to tensors
        for key in batch_obs.keys():
            batch_obs[key] = torch.tensor(batch_obs[key], dtype=torch.float).to(self.device).flatten(0, 1)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device).flatten(0, 1)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device).flatten(0, 1)
        batch_rews = torch.tensor(batch_rews, dtype=torch.float).to(self.device).flatten(0, 1)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device).flatten(0, 1)
        batch_vals = torch.tensor(batch_vals, dtype=torch.float).to(self.device).flatten(0, 1)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_vals

    def compute_rtgs(self, batch_rews):
        batch_rtgs = np.zeros_like(batch_rews)
        for env_idx in range(batch_rews.shape[1]):
            discounted_reward = 0
            for t in reversed(range(batch_rews.shape[0])):
                discounted_reward = batch_rews[t, env_idx] + self.gamma * discounted_reward
                batch_rtgs[t, env_idx] = discounted_reward
        return batch_rtgs
    
    def get_action(self, obs):
        """
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
        obs = {key: torch.from_numpy(obs[key]).to("cuda").float() for key in obs.keys()}  # Convert to torch tensors
        (mean, std), value = self.network(obs)
        dist = Independent(Normal(mean, std), 1)  # 1 = number of reinterpreted batch dims
        action = dist.sample()  # shape: (batch_size, 7)
        log_prob = dist.log_prob(action)  # shape: (batch_size,)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().squeeze(1).numpy()
    

    def evaluate(self, batch_obs, batch_acts):
        (mean, std), value = self.network(batch_obs)
        dist = Independent(Normal(mean, std), 1)  # 1 = number of reinterpreted batch dims
        log_prob = dist.log_prob(batch_acts)  # shape: (batch_size,)
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
                        max_episode_steps=1000,
                        num_envs=2,
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

