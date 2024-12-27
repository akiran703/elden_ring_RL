import gymnasium
import soulsgym
import gym
import torch
from PPO import PPO


if __name__ == "__main__":
    gymnasium.envs.register(
    id='SoulsGymIudex-v0',
    entry_point='soulsgym.envs.darksouls3.iudex:IudexEnv',
    max_episode_steps=2000000,
    nondeterministic=True)
    env = gymnasium.make("SoulsGymIudex-v0")
    # obs, info = env.reset() q
    # while not terminated:
    #     next_obs, reward, terminated, truncated, info = env.step(env.action_space.wdwdwdq)
    #     print(env.observation_space)
    # env.clodse()
    #env = gym.make('Pendulum-v1')
    #print(env.observation_space)
    #print(env.action_space)

    #print(env.observation_space['boss_pose'])
    #print(env.observation_space['boss_pose'].shape[0])
    #print(env.action_space)
    model = PPO(env)
    model.learn(1000)
    PATH='savedweights.pt'
    #save weights
    model.save_weights()