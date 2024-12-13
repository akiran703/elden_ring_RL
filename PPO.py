import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam
import numpy as np

class PPO:
    def __init__(self,env):
        # get info about the world
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        #actor and critic
        self.actor = FeedForwardNN(self.obs_dim,self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()
        
        #create our variable for the matrix
        self.cor_var = torch.full(size=(self.act_dim,),fill_value=0.5)

        #create the covariance matrix
        self.cov_mat = torch.diag(self.cor_var)

        #initalize the critic and actor optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
    
    def get_action(self,obs):
        #query the actor network for a mean action
        mean = self.actor(obs)

        #create multivariate normal distribution
        dist = MultivariateNormal(mean,self.cov_mat)

        #sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def _init_hyperparameters(self):
        #default values
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005
        
    def rollout(self):
            batch_obs = []
            batch_acts = []
            batch_log_probs = []
            batch_rews = []
            batch_rtgs = []
            batch_lens = []
            ep_rews = []
            t = 0 # Keeps track of how many timesteps we've run so far this batch

            # Keep simulating until we've run more than or equal to specified timesteps per batch
            while t < self.timesteps_per_batch:
                ep_rews = [] # rewards collected per episode

                # Reset the environment. sNote that obs is short for observation. 
                obs, _ = self.env.reset()
                done = False

                # Run an episode for a maximum of max_timesteps_per_episode timesteps
                for ep_t in range(self.max_timesteps_per_episode):
                    # If render is specified, render the environment
                   

                    t += 1 # Increment timesteps ran this batch so far

                    # Track observations in this batch
                    batch_obs.append(obs)

                    # Calculate action and make a step in the env. 
                    # Note that rew is short for reward.
                    action, log_prob = self.get_action(obs)
                    obs, rew, terminated, truncated, _ = self.env.step(action)

                    # Don't really care about the difference between terminated or truncated in this, so just combine them
                    done = terminated | truncated

                    # Track recent reward, action, and action log probability
                    ep_rews.append(rew)
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)

                    # If the environment tells us the episode is terminated, break
                    if done:
                        break

                # Track episodic lengths and rewards
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)

            # Reshape data as tensors in the shape specified in function description, before returning
            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            batch_rtgs = self.compute_reward_to_go(batch_rews)                                                              # ALG STEP 4

           

            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
            
    #calculate the Q-values
    def compute_reward_to_go(self,batch_rews):
            batch_rtgs = []

            # Iterate through each episode
            for ep_rews in reversed(batch_rews):

                discounted_reward = 0 # The discounted reward so far

                # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
                # discounted return (think about why it would be harder starting from the beginning)
                for rew in reversed(ep_rews):
                    discounted_reward = rew + discounted_reward * self.gamma
                    batch_rtgs.insert(0, discounted_reward)

            # Convert the rewards-to-go into a tensor
            batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

            return batch_rtgs
        
    #predicited values
    def evaluate(self,batch_obs,batch_acts):
        V = self.critic(batch_obs).squeeze()
        #calculate the log probs of batch actions using most recent actions in rollout
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs




    def learn(self, total_timesteps):
        #line 2 from the PP0 algo
        ts_simulated_so_far = 0
        while ts_simulated_so_far < total_timesteps:
             #collect trajectories with rollout line 3 in the PP0 algo 
             batch_obs, batch_acts, batch_logs_probs, batch_rtgs, batch_lens = self.rollout()
             
             #how many times we collected
             ts_simulated_so_far += np.sum(batch_lens)

             #calculate advantages
             V, _ = self.evaluate(batch_obs, batch_acts)

             #calculate advantage
             A_k = batch_rtgs - V.detach()

             #to deal with unstable training, we use advantage normalization to help with different dimensions
             A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
             
             #lines 6 and 7 
             #update the policy by maximizing the PPO-clip objective function
             for _ in range(self.n_updates_per_iteration):
                 # Calculate V_phi and pi_theta(a_t | s_t) 
                 V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                 #calculate ratios
                 ratios = torch.exp(curr_log_probs - batch_logs_probs)

                 #calculate surrogate losses
                 surr1 = ratios * A_k
                 surr2 = torch.clamp(ratios,1-self.clip, 1 + self.clip) * A_k

                 actor_loss = (-torch.min(surr1, surr2)).mean()

                 critic_loss = nn.MSELoss()(V, batch_rtgs)

                 # Calculate gradients and perform backward propagation for actor network
                 self.actor_optim.zero_grad()
                 actor_loss.backward()
                 self.actor_optim.step()
        print('done')




                 







