import torch 

from torch.distributions import MultivariateNormal
from network import FeedForwardNN

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
    #collect trajectories with rollout line 3 in the PP0 algo 
    def rollouts(self):
        batch_obs = [] # batch observation 
        batch_acts = [] #batch action
        batch_log_probs = [] # log prob of each action 
        batch_rews = [] # batch rewards
        batch_rtgs = [] # batch rewards to go
        batch_lens = [] #episodic lengths in batch

        #number of timesteps run for this batch
        t = 0 
        while t < self.timesteps_per_batch:
            #rewards this epi
            ep_rews = []
            
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                #increment timesteps ran this batch 
                t+=1

                #observation collecting
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done = self.env.step(action)

                #collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
                
                #collect episodic length and rewards
                batch_lens.append(ep_t+1)
                batch_rews.append(ep_rews)

                #reshape data as tensors 
                batch_obs = torch.tensor(batch_obs,dtype=torch.float)
                batch_acts = torch.tensor(batch_acts,dtype=torch.f)

    def learn(self, total_timesteps):
        #line 2 from the PP0 algo
        ts_simulated_so_far = 0
        while ts_simulated_so_far < total_timesteps:





