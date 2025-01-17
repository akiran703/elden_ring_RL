import gymnasium
import soulsgym
import gym
import torch
from PPO import PPO
import optuna
import os


# if __name__ == "__main__":
#     gymnasium.envs.register(
#     id='SoulsGymIudex-v0',
#     entry_point='soulsgym.envs.darksouls3.iudex:IudexEnv',
#     max_episode_steps=2000000,
#     nondeterministic=True)
#     env = gymnasium.make("SoulsGymIudex-v0")

#     actor_path = "./savedactor.pt"
#     critic_path = "./savedcritic.pt"

#     model = PPO(env)
#     model.learn(10000)
#     PATH='savedweights.pt'       
#     #save weights
#     model.save_weights()


def objective(trial):
    gymnasium.envs.register(
    id='SoulsGymIudex-v0',
    entry_point='soulsgym.envs.darksouls3.iudex:IudexEnv',
    max_episode_steps=2000000,
    nondeterministic=True)
    env = gymnasium.make("SoulsGymIudex-v0")
    model = PPO(env, trial = trial)
    total_timesteps = 5000
    mean_reward = model.learn(total_timesteps)
    output_dir = f"trial_{trial.number}_results"  # Create a directory specific to the trial
    model.save_trial_plots(output_dir,trial.number)
    output_parent_dir = '.\fullcollections'
    os.path.join(output_parent_dir,output_dir)
    return mean_reward
def run_optuna_study():

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = 10)

        print("Best trial:", study.best_trial.value)
        print("Best Params:", study.best_trial.params)

if __name__ == '__main__':
    run_optuna_study()