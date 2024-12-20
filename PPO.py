import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam
import numpy as np
from torch.distributions.categorical import Categorical
from scipy import stats


class PPO:
    def __init__(self,env):
        # get info about the world
        self.env = env
        self.obs_dim = 8
        self.act_dim = 20

        

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
        #mean = self.actor(obs)
        #print('printing the mean')
        #print(mean)
        #print(mean.shape)

        #create multivariate normal distribution
        #dist = Categorical(logits=mean)

        #sample an action from the distribution and get its log prob
        #action = dist.sample()
        #print(action)
        #log_prob = dist.log_prob(action)

        # Query the actor network for logits (raw predictions for each action)
        logits = self.actor(obs)
        
        # Convert logits to probabilities using softmax
        prob = torch.softmax(logits, dim=-1)
        
        # Calculate the variance of the probabilities across actions
        action_variance = torch.var(prob).item()  # Get variance as a scalar
        
        # Log the variance to track how spread out the probabilities are
        #print(f"Action Probability Variance: {action_variance}")
        
        # Create Categorical distribution with logits
        dist = Categorical(logits=logits)
        
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob



        

        return action, log_prob

    def _init_hyperparameters(self):
        #default values
        self.timesteps_per_batch = 48000
        self.max_timesteps_per_episode = 16000
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005
        self.num_minibatches = 10
        self.ent_coef = 0.1
        self.max_grad_norm = 0.5
        self.lam = 0.98
        self.target_kl = 0.02
        
    def rollout(self):
            batch_obs = []
            batch_acts = []
            batch_log_probs = []
            batch_rews = []
            batch_rtgs = []
            batch_lens = []
            
            batch_vals = []
            batch_dones = []
            actioncount = [0] * 20

            
            #we clear 
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # values collected per episode
            ep_dones = [] #dones collected per episode
            t = 0 # Keeps track of how many timesteps we've run so far this batch

            # Keep simulating until we've run more than or equal to specified timesteps per batch
            while t < self.timesteps_per_batch:
                ep_rews = [] # rewards collected per episode
                ep_vals = [] # values collected per episode
                ep_dones = [] #dones collected per episode


                # Reset the environment. sNote that obs is short for observation. 
                obs, _ = self.env.reset()
                done = False

                # Run an episode for a maximum of max_timesteps_per_episode timesteps
                for ep_t in range(self.max_timesteps_per_episode):
                    # If render is specified, render the environment
                    ep_dones.append(done)

                    t += 1 # Increment timesteps ran this batch so far

                    # Track observations in this batch
                    #print(obs)
                    #boss_animation, boss_animation_duration, boss_hp, boss_max_hp, boss_pose, camera_pose, lock_on, phase, player_animation, player_animation_duration, player_hp, player_max_hp, player_max_sp, player_pose, player_sp = obs.values()
                    #print(boss_animation)

                    boss = torch.tensor([obs['boss_animation'], obs['boss_animation_duration'], obs['boss_hp'], 0])
                    player = torch.tensor([obs['player_animation'], obs['player_animation_duration'], obs['player_hp'], obs['player_sp']])
                    #boss_pose = torch.tensor(obs['boss_pose'])
                    #player_pose = torch.tensor(obs['player_pose'])

                    #print(boss.shape)
                    #print(player.shape)
                    #print(boss_pose.shape)
                    #print(player_pose.shape)


                    #obs = torch.cat((boss, player, boss_pose, player_pose),0)
                    obs = torch.cat((boss, player),0)

                    #print(obs)
                    #print(obs.shape)
                    
                    batch_obs.append(obs)

                    # Calculate action and make a step in the env. 
                    # Note that rew is short for reward.
                    action, log_prob = self.get_action(obs)
                    val = self.critic(obs)
                    #print('we are printing the action')
                    #print(action.item())
                    
                    actioncount[action.item()] += 1
                    obs, rew, terminated, truncated, _ = self.env.step(action.item())
                    print('printing the reward')
                    print(rew)


                    # Don't really care about the difference between terminated or truncated in this, so just combine them
                    done = terminated or truncated

                    # Track recent reward, values, action, and action log probability
                    ep_rews.append(rew)
                    ep_vals.append(val.flatten())
                    batch_acts.append(action.item())
                    batch_log_probs.append(log_prob)

                    # If the environment tells us the episode is terminated, break
                    if done:
                        break

                # Track episodic lengths and rewards
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)


            # Reshape data as tensors in the shape specified in function description, before returning
            # ALG STEP 4
            batch_rtgs = self.compute_reward_to_go(batch_rews)                                                             
            batch_obs = torch.stack(batch_obs)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            #print('the problem')
            #print(batch_acts.shape)
            #print('yo yo')

            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_dones, batch_rews, batch_vals, actioncount 
            
    #calculate the Q-values
    def compute_reward_to_go(self,batch_rews):
            batch_rtgs = []

            # Iterate through each episodewdwdwdwdwdqp
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
        #print(type(batch_acts))
        V = self.critic(batch_obs).squeeze()
        
        #calculate the log probs of batch actions using most recent actions in rollout
        #print(batch_obs.shape)
        mean = self.actor(batch_obs)
        #print(mean)
        
        dist = Categorical(logits=mean)
        #print('we are in inside evaluate')
        #print(dist)
        #print(batch_acts.shape)
        log_probs = dist.log_prob(batch_acts)
        #print('hello')
        return V, log_probs, dist.entropy()
    
    def calculate_gae(self,rewards,values,dones):
        #store computed advantagtes for each timestep
        batch_advantages = []

        #iterate over the rewards, dones and values at each episode
        for ep_rews,ep_vals,ep_dones in zip(rewards,values,dones):
            #store current episode
            advantages = []
            #initialize the last computed advantage
            last_advantage = 0

            #calcualte the episode advantage in reverse order
            for t in reversed(range(len(ep_rews))):
                if (t+1) < len(ep_rews):
                    #calculate the temporal differences
                    #difference between the observered rewards and estimated values of states
                    #it captures the immediate impact of an action and provides information about how much better or worse the actual outcome was compared to the expected outcome
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    #last timestep
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list
            
            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)






    def learn(self, total_timesteps):
        #line 2 from the PP0 algo
        ts_simulated_so_far = 0
        while ts_simulated_so_far < total_timesteps:
             #collect trajectories with rollout line 3 in the PP0 algo 
             batch_obs, batch_acts, batch_logs_probs, batch_rtgs, batch_lens,batch_dones, batch_rews, batch_vals, dataforgraph = self.rollout()
             
             #calculate advantage with GAE
             A_k = self.calculate_gae(batch_rews,batch_vals,batch_dones)
             V = self.critic(batch_obs).squeeze()
             batch_rtgs = A_k + V.detach()

            


             #how many times we collected
             ts_simulated_so_far += np.sum(batch_lens)

            #plot action distribution
             maxaction =  max(dataforgraph)
             print(dataforgraph.index(maxaction) + 1)


             
             #doing minibatch setup
             step = batch_obs.size(0)
             inds = np.arange(step)
             minibatch_size = step // self.num_minibatches

             #lines 6 and 7 
             #update the policy by maximizing the PPO-clip objective function
             for _ in range(self.n_updates_per_iteration):
                 #implement learning rate annealing
                
                 frac = (ts_simulated_so_far - 1.0) / total_timesteps
                 new_lr = self.lr * (1.0 - frac)
                 new_lr = max(new_lr, 0.0)
                 self.actor_optim.param_groups[0]["lr"] = new_lr
                 self.critic_optim.param_groups[0]["lr"] = new_lr


                 np.random.shuffle(inds)
                 for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_logs_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) 
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)


                    logratios = curr_log_probs - mini_log_prob
                    #calculate ratios
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios-1) - logratios).mean()


                    #calculate ratios
                    #ratios = torch.exp(curr_log_probs - batch_logs_probs)

                    #calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    # to ensure balance of exploration and exploitation, we incorporate entropy into the loss function, resulting in more versatile actions taking place
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()


                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()
                 if approx_kl > self.target_kl:
                     break
        print('done')




                 







