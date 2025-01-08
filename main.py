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

    actor_path = "./savedactor.pt"
    critic_path = "./savedcritic.pt"

    model = PPO(env)
    model.learn(10000)
    PATH='savedweights.pt'       
    #save weights
    model.save_weights()