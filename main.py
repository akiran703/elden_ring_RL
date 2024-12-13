import gymnasium
import soulsgym
import gym
from PPO import PPO


if __name__ == "__main__":
    #env = gymnasium.make("SoulsGymIudex-v0")
    # obs, info = env.reset() 
    # terminated = False
    # while not terminated:
    #     next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    #     print(env.observation_space)
    # env.close()
    env = gym.make('Pendulum-v1')
    print(env.observation_space)
    model = PPO(env)
    model.learn(10000)
