import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
       
class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        ### TODO: write correct input size
        self.layers = nn.Sequential(
            nn.Linear(..., 512), nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self,state,previous_state):

        ### flatten state (keep batch dim)
        state = state.reshape(state.shape[0],-1)
        previous_state = previous_state.reshape(previous_state.shape[0],-1)

        ### inputs: current state and previous state
        combined = torch.cat([state,previous_state],dim=-1)
        logits = self.layers(combined)
        
        ### sample from logits to select one action
        c = torch.distributions.Categorical(logits=logits)
        action = c.sample().squeeze(-1)

        ### this is the (log) probability of selecting that action
        action_log_prob = c.log_prob(action)

        ### TODO: fill the outputs
        output = {'logits': ...,
                  'action': ...,
                  'action_log_prob': ... 
                  }

        return output
