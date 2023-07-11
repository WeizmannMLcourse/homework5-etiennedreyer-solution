import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,logits,action,discounted_rewards):
    
        ### (log) probabilities for all possible next actions
        all_action_log_probs = F.log_softmax(logits,dim=1)

        ### the probability of the actual action that was selected by sampling 
        sel_action_log_prob = torch.gather(all_action_log_probs, dim=1, index=action.unsqueeze(1))
        sel_action_log_prob = sel_action_log_prob.view(-1)

        ### TODO: negative log probability times discounted reward
        policy_loss = ...
        policy_loss = torch.mean(policy_loss)
        
        return policy_loss