import os
import gymnasium as gym
import numpy as np
import torch
from dataset import GamesMemoryBank
from model import PolicyNetwork

def evaluate(n_games=10):
	
    net = PolicyNetwork()
    net.load_state_dict(torch.load('trained_model.pt'))
    net.eval()
    
    env = gym.make("ALE/Pong-v5")
    env.reset()

    memory_bank = GamesMemoryBank()
    points = memory_bank.play_games(n_games,env,net)

    avg_points = np.mean(points)

    return avg_points

def test():
	
	assert os.path.exists('trained_model.pt'), "Need to upload trained_model.pt"

	avg_points = evaluate()

	assert (avg_points > 8) , "Need to have average score above 8. You got {}".format(avg_points)
	 
if __name__ == '__main__':
	test()