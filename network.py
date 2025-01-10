import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim,out_dim):
        super().__init__()
        self.policy_layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
             nn.Linear(64, out_dim)
        )
 

    def forward(self,obs):
           policy = self.policy_layers(obs)
           return policy
    
