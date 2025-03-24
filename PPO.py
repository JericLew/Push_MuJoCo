# Description: Implementing PPO-Clip based on OpenAI pseudocode
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

# Inputs to PPO Clip Algorithm: Network, Gym Environment, Hyperparameters
# Outputs: Stores training progress graphs 

import torch
from torch.distributions import MultivariateNormal


from network import FeedForwardNN # PPO Network
from push_nn import PushNN

class PPOAgent():
    def __init__(self, env):
        self.env = env

        # Extract Observation and Action Space from environment
        self.obs_dim = env.observation_space.shape[0] 
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95 # Discount Factor

    def learn(self, total_timesteps):
        t_so_far = 0 # timesteps so far

        while t_so_far < total_timesteps: # ALG step 2
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

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

            # Store rewards per episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range (self.max_timesteps_per_episode):
                
                # Increment timesteps ran so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # +1 because timesteps start from 0
            batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)         # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
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
        # Query the actor network for a mean action (center of Gaussian distribution)
        mean = self.actor(obs)

        # Creates a Gaussian Distribution centered around mean
        # self.cov_mat is a covariance matrix
        dist = MultivariateNormal(mean, self.cov_mat)

        # Samples an action from the distribution 
        action = dist.sample()

        # Computes the log probability of a selected action (for PPO policy updates)
        log_prob = dist.log_prob(action)

        # Returns sampled action and log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()


        

