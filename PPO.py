import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam
import numpy as np
from torch.distributions.categorical import Categorical
from scipy import stats
import statistics
import matplotlib.pyplot as plt


class PPO:
    def __init__(self,env, actor_path = None, critic_path= None):
        # get info about the world
        self.env = env
        #8
        self.obs_dim = self.get_observation_space_size()  # Dynamically get obs dim
        #20
        self.act_dim = self.env.action_space.n #Dynamically get act dim

        #store reward and most used action
        self.globalreward = []
        self.globalaction = []

        # Track training progress
        self.episode_rewards = []
        self.mean_rewards = []
        self.action_histories = []
        self.episode_lengths = []

         # Track loss and KL divergence
        self.actor_losses = []
        self.critic_losses = []
        self.kl_divergences = []
        

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

        #if paths are passed then load the weights
        if actor_path and critic_path:
            self.load_weights(actor_path,critic_path)

    def load_weights(self, actor_path, critic_path):
        print('loading weights')
        try:
            self.actor.load_state_dict(torch.load(actor_path, weights_only=True))
            self.critic.load_state_dict(torch.load(critic_path, weights_only=True))
            print("Loaded weights successfully")
        except Exception as e:
            print(f"Could not load weights: {e}")
    
    def get_observation_space_size(self):
      #gets total observation length
       obs, _ = self.env.reset()
       total_obs_length = 0
       for value in obs.values():
           if isinstance(value, np.ndarray):
              total_obs_length += value.size
           else:
              total_obs_length += 1
       return total_obs_length
    
    def get_action(self,obs, epsilon):
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
        
        #if random is greater than epsilon_threshold take a random action
        if np.random.rand() < epsilon:
                action = np.random.choice(self.act_dim)
                action = torch.tensor(action)
                log_prob = torch.tensor(1, dtype=torch.float)
        # Convert logits to probabilities using softmax
        else:
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

        return action.item(), log_prob



    def _init_hyperparameters(self):
        #default values
        self.timesteps_per_batch = 480
        self.max_timesteps_per_episode = 160
        self.gamma = 0.95
        self.n_updates_per_iteration = 10
        self.clip = 0.3
        self.lr = 0.01
        self.num_minibatches = 10
        self.ent_coef = 0.1  # Reduced entropy coefficient
        self.max_grad_norm = 0.5
        self.lam = 0.98
        self.target_kl = 0.05
        self.exploration_start = 0.7
        self.exploration_end = 0.02
        self.exploration_decay = 5000 # Decay rate
        

    def preprocess_observation(self, obs):
        # Process observation dictionary into a single tensor
        processed_obs = []
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                processed_obs.append(torch.tensor(value.flatten(), dtype=torch.float32))
            else:
                processed_obs.append(torch.tensor([value], dtype=torch.float32))
        return torch.cat(processed_obs, dim=0)
      
    def rollout(self,epsilon_threshold=0.0):
            batch_obs = []
            batch_acts = []
            batch_log_probs = []
            batch_rews = []
            batch_rtgs = []
            batch_lens = []
            
            batch_vals = []
            batch_dones = []
            actioncount = [0] * self.act_dim
            trackreward = []

            
            #we clear 
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # values collected per episode
            ep_dones = [] #dones collected per episode
            t = 0 # Keeps track of how many timesteps we've run so far this batch
            episode_actions_temp = [] #temporary list for logging actions
            # Keep simulating until we've run more than or equal to specified timesteps per batch
            while t < self.timesteps_per_batch:
                ep_rews = [] # rewards collected per episode
                ep_vals = [] # values collected per episode
                ep_dones = [] #dones collected per episode


                # Reset the environment. sNote that obs is short for observation. 
                obs, _ = self.env.reset()
                prebosshealth = obs['boss_hp']
                prehealth = obs["player_hp"]
                
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

                    #boss = torch.tensor([obs['boss_animation'], obs['boss_animation_duration'], obs['boss_hp'], 0])
                    #player = torch.tensor([obs['player_animation'], obs['player_animation_duration'], obs['player_hp'], obs['player_sp']])

                    #boss_pose = torch.tensor(obs['boss_pose'])
                    #player_pose = torch.tensor(obs['player_pose'])

                    #print(boss.shape)
                    #print(player.shape)
                    #print(boss_pose.shape)
                    #print(player_pose.shape)


                    #obs = torch.cat((boss, player, boss_pose, player_pose),0)
                    #obs = torch.cat((boss, player),0)

                    #print(obs)
                    #print(obs.shape)

                    obs_tensor = self.preprocess_observation(obs)
                    batch_obs.append(obs_tensor)

                    
                    #batch_obs.append(obs)

                    # Calculate action and make a step in the env. 
                    # Note that rew is short for reward.
                    action, log_prob = self.get_action(obs_tensor,epsilon_threshold)
                    val = self.critic(obs_tensor)
                    #print('printing val type')
                    #print(type(val))
                    #print('we are printing the action')
                    #print(action.item())
                    
                    
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    #Reward Shaping Constants
                    BOSS_DAMAGE_REWARD = 1000
                    PLAYER_DAMAGE_PENALTY = -100
                    NO_DAMAGE_REWARD = 5
                    NO_BOSS_DAMAGE_PENALTY = -10
                    BOSS_HEALTH_REWARD = 1
                    rew = 0

                    #we will calculate the rew based on the boss hp 
                    didbosshealthchange =  prebosshealth - obs['boss_hp'] 
                    #Reward for damaging the boss, and an incentive to deal damage. 
                    # if didbosshealthchange != 0:
                    #             # Reward for Damaging the boss
                    #     rew += (BOSS_DAMAGE_REWARD * didbosshealthchange)/1000
                    #     rew = int(rew.item()) #convert to int

                    #     if obs["player_hp"] < prehealth:
                    #         # Penalty for getting hit while damaging the boss
                    #         rew += PLAYER_DAMAGE_PENALTY
                    # elif obs["player_hp"] >= prehealth:
                    #     # Reward for surviving if the boss is not hit
                    #     #rew += NO_DAMAGE_REWARD
                    #     rew-=1
                    # else:
                    #     # Penalty for getting hit if the boss isn't hit
                    #     rew += NO_BOSS_DAMAGE_PENALTY
                    if didbosshealthchange != 0:  # If boss took damage
                            # Reward for damaging the boss based on a normalized value
                            rew +=  int((BOSS_DAMAGE_REWARD * didbosshealthchange)/1000)

                            if obs["player_hp"] < prehealth: # if player gets damaged while hitting the boss
                                rew += PLAYER_DAMAGE_PENALTY # Add penalty for getting hit

                    elif obs["player_hp"] <= 0:
                        #If the player took damage and did not damage the boss
                            rew += NO_BOSS_DAMAGE_PENALTY  # Add penalty for getting hit when no boss damage


                        # Reward for moving boss health closer to 0
                    rew += int(BOSS_HEALTH_REWARD * (obs['boss_max_hp']-obs["boss_hp"])/1000)
                    

                     
                    print(f"Action: {action}, Reward: {rew}, boss health change: {didbosshealthchange}")




                    
    
                    # Don't really care about the difference between terminated or truncated in this, so just combine them
                    # Update for next step
                    prehealth = obs["player_hp"]
                    prebosshealth = obs['boss_hp']
                    done = terminated or truncated

                    # Track recent reward, values, action, and action log probability
                    ep_rews.append(rew)
                    #print('ep_rew type')
                    #print(type(ep_rews))
                    ep_vals.append(val.flatten())
                    #print('ep_vals type')
                    #print(type(ep_vals))
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)
                    episode_actions_temp.append([int(action)])
                    # If the environment tells us the episode is terminated, break
                    if done:
                        break

                # Track episodic lengths and rewards
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)
            

            #taking average of batch_rews
            trackreward.append(statistics.mean([statistics.mean(sublist) for sublist in batch_rews]))


            # Reshape data as tensors in the shape specified in function description, before returning
            # ALG STEP 4
            batch_rtgs = self.compute_reward_to_go(batch_rews)                                                             
            batch_obs = torch.stack(batch_obs)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
           

            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_dones, batch_rews, batch_vals, actioncount, trackreward, episode_actions_temp
            
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
        V = self.critic(batch_obs)
        print(f"Critic Value: {V.mean()}")
        
        #calculate the log probs of batch actions using most recent actions in rollout
        #print(batch_obs.shape)
        logits = self.actor(batch_obs)
        #print(mean)
        
        print(f"Action Probs: {torch.softmax(logits, dim=-1).mean(dim=0)}")

        dist = Categorical(logits=logits)
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

    def save_weights(self):
        print('saving weights')
        P1 = 'savedactor.pt'
        P2 = 'savedcritic.pt'
        torch.save(self.actor.state_dict(), P1 )
        torch.save(self.critic.state_dict(),P2)

    def plot_training_info(self):
                # Calculate mean episode rewards
            episode_indices = np.arange(1, len(self.episode_rewards) + 1)

            # Calculate mean rewards using a rolling average
            window_size = 20
            rolling_mean_rewards = np.convolve(self.episode_rewards, np.ones(window_size) / window_size, mode='valid')

            # Plotting mean episode rewards
            plt.figure(figsize=(10, 6))
            plt.plot(episode_indices[window_size - 1:], rolling_mean_rewards, label='Mean Episode Reward (Rolling Average)')
            plt.xlabel('Episode')
            plt.ylabel('Mean Reward')
            plt.title('Mean Episode Rewards Over Training')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plotting Action Distribution
            action_frequency = [0] * self.act_dim
            # for each episode we calculate how many times each action has been made and keep a count
            for action_list in self.action_histories:
                for action_index in action_list:
                    action_frequency[int(action_index)] += 1

            total_count = sum(action_frequency)
            

            if total_count > 0:
                #calculate action percentages
                action_percentages = [count / total_count * 100 for count in action_frequency]


                    #plot bar char
                plt.figure(figsize=(10, 6))
                plt.bar(range(self.act_dim), action_percentages)
                plt.xlabel("Action")
                plt.ylabel("Percentage of action use")
                plt.title("Action Percentage Distribution Across Training")
                plt.xticks(range(self.act_dim))
                plt.grid(axis='y')
                plt.show()
            else:
                print("No actions were taken, cannot plot action distribution")

            # Plotting Training Progress
            plt.figure(figsize=(10, 6))
            plt.plot(self.globalreward)
            plt.xlabel('Iteration')
            plt.ylabel('Reward')
            plt.title('Reward Over Training')
            plt.grid(True)
            plt.show()

    def plot_loss_kl(self):
            # Plotting Actor and Critic Losses
            plt.figure(figsize=(10, 6))
            plt.plot(self.actor_losses, label='Actor Loss')
            plt.plot(self.critic_losses, label='Critic Loss')
            plt.xlabel('Update Iteration')
            plt.ylabel('Loss')
            plt.title('Actor and Critic Loss Over Training')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plotting KL Divergence
            plt.figure(figsize=(10, 6))
            plt.plot(self.kl_divergences, label='KL Divergence')
            plt.xlabel('Update Iteration')
            plt.ylabel('KL Divergence')
            plt.title('KL Divergence Over Training')
            plt.legend()
            plt.grid(True)
            plt.show()

    def learn(self, total_timesteps):
        #line 2 from the PP0 algo
        ts_simulated_so_far = 0
        self.action_histories_temp = []
        random_action_steps = 1000
        while ts_simulated_so_far < total_timesteps:
             epsilon_threshold = self.exploration_end + (self.exploration_start - self.exploration_end) * np.exp(-1 * ts_simulated_so_far / self.exploration_decay)
             #collect trajectories with rollout line 3 in the PP0 algo 
             if ts_simulated_so_far < random_action_steps:
                batch_obs, batch_acts, batch_logs_probs, batch_rtgs, batch_lens,batch_dones, batch_rews, batch_vals, dataforgraph,dataforreward,episode_actions_temp = self.rollout(1.0)
             else:
                batch_obs, batch_acts, batch_logs_probs, batch_rtgs, batch_lens,batch_dones, batch_rews, batch_vals, dataforgraph,dataforreward,episode_actions_temp = self.rollout(epsilon_threshold)
             
             #calculate advantage with GAE
             A_k = self.calculate_gae(batch_rews,batch_vals,batch_dones)
             V = self.critic(batch_obs).squeeze()
             batch_rtgs = A_k + V.detach()

            


             #how many times we collected
             ts_simulated_so_far += np.sum(batch_lens)

             # Save episode length, reward and most used action
             self.episode_rewards.extend([sum(r) for r in batch_rews])
             self.episode_lengths.extend(batch_lens)
             
             self.globalreward.extend(dataforreward)
             
             self.action_histories_temp.extend(episode_actions_temp)
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
                 
                 #print(f"Current LR: {self.actor_optim.param_groups[0]['lr']}")

                 np.random.shuffle(inds)
                 for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_logs_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    if torch.all(mini_advantage == 0):
                        print('the advantage is zero')
                        continue

                    # Calculate V_phi and pi_theta(a_t | s_t) 
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    print(f"mini_advantage: {mini_advantage.mean()}")
                    # print(f"curr_log_probs: {curr_log_probs.mean()}")
                    # print(f"mini_log_prob: {mini_log_prob.mean()}")
                    # print(f"entropy: {entropy.mean()}")


                    logratios = curr_log_probs - mini_log_prob
                    #calculate ratios
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios-1) - logratios).mean()

                    # print(f"ratios {ratios.mean()}")


                    #calculate ratios
                    #ratios = torch.exp(curr_log_probs - batch_logs_probs)

                    #calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    #surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                    surr2 = ratios * mini_advantage
                    if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                        print("NaN surrogate loss detected")


                    # print(f"surr1: {surr1.mean()}")
                    # print(f"surr2: {surr2.mean()}")
                    
                    

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    critic_loss = nn.MSELoss()(V.flatten(), mini_rtgs)

                    #print(f"Actor Loss Before Entropy: {actor_loss}")
                    

                    # Entropy Regularization
                    # to ensure balance of exploration and exploitation, we incorporate entropy into the loss function, resulting in more versatile actions taking place
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    #print(f"Actor Loss After Entropy: {actor_loss}")

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    for name, param in self.actor.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad.abs().mean()):
                                    print(f"Actor Parameter {name} Gradient: NaN")
                            #else:
                                #print(f"Actor Parameter {name} Gradient: {param.grad.abs().mean()}")
                        else:
                                print(f"Actor Parameter {name} Gradient: None")
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()


                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                 #store metrics
                 self.actor_losses.append(actor_loss.item())
                 self.critic_losses.append(critic_loss.item())
                 self.kl_divergences.append(approx_kl.item())

                 if approx_kl > self.target_kl:
                     break
        self.action_histories = self.action_histories_temp
        print('done')
        self.plot_training_info()
        self.plot_loss_kl()

        
        




                 







