# Description: Implementing PPO-Clip based on OpenAI pseudocode
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

# Inputs to PPO Clip Algorithm: Network, Gym Environment, Hyperparameters
# Outputs: Stores training progress graphs 

import torch
import numpy as np
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.distributions import Normal, Independent

from network import FeedForwardNN # PPO Network
from push_nn import PushNN

class PPOAgent():
    def __init__(self, env, network):
        # Initialize Hyperparameters
        self._init_hyperparameters()

        self.logger = {}
        
        # Initialize Environment
        self.env = env
        
        # Initialize Actor and Critic Networks
        self.network = network

        self.network_optim = Adam(self.network.parameters(), lr=self.lr)

        # # Extract Observation and Action Space from environment
        # self.env = env
        # self.obs_dim = env.observation_space.shape[0] 
        # self.act_dim = env.action_space.shape[0]

        # # Initialize Actor and Critic Networks
        # self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        # self.critic = FeedForwardNN(self.obs_dim, 1)

        # # Initialize Actor Optimizer 
        # self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        # self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # self.cov_mat = torch.diag(self.cov_var)


    def _init_hyperparameters(self):
        self.timesteps_per_batch = 10000        # timesteps per batch
        self.max_timesteps_per_episode = 1000   # timesteps per episode
        self.gamma = 0.95                       # Discount Factor
        self.n_updates_per_iteration = 5        # number of epochs per iteration
        self.clip = 0.2                         # PPO clip parameter (recommended by paper)
        self.lr = 0.005                         # Learning Rate

    def learn(self, total_timesteps):
        t_so_far = 0 # timesteps so far

        while t_so_far < total_timesteps: # ALG step 2
            batch_states, batch_images, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate V from critic network (value function estimate) for each state in batch_obs
            # V is needed to compute the advantage function; It estimates how good a state is 
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG Step 5
            # Calculate advantage
            # If A_k is positive, action was better than expected; If A_k negative, action was worse than expected
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            # Advantage values can have high variance, normalization helps stabilize training
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            t_so_far += np.sum(batch_lens)

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

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph = True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()





    def rollout(self):
        """
            Responsible for collecting data by interacting with the simulation environment.
            PPO is on-policy, hence needs fresh data from current policy at each training iteration

            Returns:
            	batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """

        batch_obs = [] # batch observations, should be feature vectors from CNN (push_nn)
        batch_acts = [] # batch actions: stores action taken at each step
        batch_log_probs = [] # log probabilities of each action 
        batch_rews = [] # batch rewards: stores reward received at each timestep
        batch_rtgs = [] # batch rewards-to-go: stores the future reward from each step onward (long-term)
        batch_lens = [] # number of episodes per batch

        t = 0 # keeps track of how many timesteps we have run so far for this batch

        while t < self.timesteps_per_batch:
            print("Timesteps so far: ", t)

            # Store rewards per episode
            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range (self.max_timesteps_per_episode):
                
                # Increment timesteps ran so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                done = terminated | truncated

                # Collect reward, action, and log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # +1 because timesteps start from 0
            batch_rews.append(ep_rews)

        batch_states = [obs["state"] for obs in batch_obs]
        batch_images = [obs["image"] for obs in batch_obs]

		# Reshape data as tensors in the shape specified in function description, before returning
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float)
        batch_images = torch.tensor(np.array(batch_images), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)         # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_states, batch_images, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def compute_rtgs(self, batch_rews):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""

        batch_rtgs = []

        # Iterate through each epsiode backwards to maintain same order
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 

            for rew in reversed (ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

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
        (mean, std), value = self.network(obs)

        # Create a multivariate distribution assuming diagonal covariance (independent Gaussians)
        dist = Independent(Normal(mean, std), 1)  # 1 = number of reinterpreted batch dims

        action = dist.sample()  # shape: (batch_size, 7)
        log_prob = dist.log_prob(action)  # shape: (batch_size,)
        return action.detach().cpu().numpy(), log_prob.detach()

        # # Query the actor network for a mean action (center of Gaussian distribution)
        # mean = self.actor(obs)

        # # Creates a Gaussian Distribution centered around mean
        # # self.cov_mat is a covariance matrix
        # dist = MultivariateNormal(mean, self.cov_mat)

        # # Samples an action from the distribution 
        # action = dist.sample()

        # # Computes the log probability of a selected action (for PPO policy updates)
        # log_prob = dist.log_prob(action)

        # # Returns sampled action and log probability of that action in our distribution
        # return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and log probs log_probs
        return V, log_probs

        

if __name__ == "__main__":
    # Example usage of the PPOAgent class
    # Initialize the environment and agent

    import gymnasium as gym
    from push_custom import PickPlaceCustomEnv

    xml_path = "franka_emika_panda/scene_push.xml"
    env = PickPlaceCustomEnv(xml_path, render_mode="camera")

    state_dim = env.observation_space["state"].shape[0]
    image_dim = env.observation_space["image"].shape
    action_dim = env.action_space.shape[0]
    print("State Space: ", state_dim)
    print("Image Space: ", image_dim)
    print("Action Space: ", action_dim)
    network = PushNN(state_dim=state_dim, image_dim=image_dim, action_dim=action_dim)

    rl_agent = PPOAgent(env, network)
    rl_agent.learn(10000)

