import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

### crop away the score chart, downsample, and convert RGB image to binary
class PreProcess(nn.Module):

    def __init__(self):

        super().__init__()
        pass

    def forward(self, img):

        if img is None:
            return torch.zeros(75, 80)

        img = img[35:185]  # crop 
        img = img[::2,::2,0] # downsample by factor of 2.
        img[img == 144] = 0  # erase background (background type 1)
        img[img == 109] = 0  # erase background (background type 2)
        img[img != 0] = 1    # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively

        img = torch.from_numpy(img.astype(np.float32)).float()

        return img


class GamesMemoryBank(Dataset):
	
	def __init__(self,do_preprocess=True):

		self.gamma = 0.99
		self.clear_memory()
		self.do_preprocess = do_preprocess
		self.preprocess = PreProcess()


	def clear_memory(self):

		self.state_history = []
		self.previous_state_history = []
		self.action_history = []
		self.action_log_prob_history = []
		self.reward_history = []


	def add_event(self, state, previous_state, action, action_log_prob, reward):

		self.state_history.append(state)
		self.previous_state_history.append(previous_state)
		self.action_history.append(action)
		self.action_log_prob_history.append(action_log_prob)
		self.reward_history.append(reward)


	def compute_reward_history(self):

		R = 0
		self.discounted_rewards = []

		### step backwards through reward history
		for r in self.reward_history[::-1]:

			### The reward is only nonzero if the network scored or lost a point (r=1 or -1):
			###     e.g. [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...,0, -1, 0, 0, 0, 0, 0, 0, 0, 0]
			### At those r!=0 timesteps we reset the discounted reward to 0
			if r != 0: 
				R = 0

			### TODO: Discounted reward is the reward at this step plus the discounted reward from the next step.
			### Since we are stepping backwards through steps, R at the "next step" corresponds to the current value of R.
			### Take that and multiply by the discount factor.
			### Finally, add it to the reward at this timestep.
			### The answer is the new discounted reward (i.e. the one at one timestep back).
			R = ...
			
			### "pre-pend" the discounted reward to the list of discounted rewards
			self.discounted_rewards.insert(0, R)
		
		self.discounted_rewards = torch.FloatTensor(self.discounted_rewards)

		### normalize discounted rewards (helpful for training stability)
		self.discounted_rewards = (self.discounted_rewards - self.discounted_rewards.mean()) / self.discounted_rewards.std()


	def __len__(self):
	   
		return len(self.state_history)


	def __getitem__(self, idx):
		
		state = self.state_history[idx]
		previous_state = self.previous_state_history[idx]

		if self.do_preprocess:
			state = self.preprocess(state)
			previous_state = self.preprocess(previous_state)

		action = torch.tensor( self.action_history[idx] )
		action_log_prob = torch.tensor( self.action_log_prob_history[idx] )
		reward = torch.tensor( self.reward_history[idx] )

		discounted_reward = self.discounted_rewards[idx]
		
		return state, previous_state, action, action_log_prob, reward, discounted_reward


	def play_games(self,n_games,env,net,n_steps=190000):
	
		points_in_all_games = []

		for _game_i in tqdm( range(n_games) ):

			### start new game
			state, _info = env.reset()
			previous_state = None
			points_in_game = 0

			for _t in range(n_steps):

				### preprocess state so the network can handle it
				state_preprocessed = self.preprocess(state).view(-1).unsqueeze(0)
				previous_state_preprocessed = self.preprocess(previous_state).view(-1).unsqueeze(0)

				### get action from policy network
				with torch.no_grad():
					output = net(state_preprocessed,previous_state_preprocessed)
					action = int(output['action'].cpu().numpy())
					action_log_prob = float(output['action_log_prob'].detach().cpu().numpy())

				### take a step in the environment
				### Note: shift action by 2 to convert to 6-dimensional action space:
				### ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
				new_state, reward, done, _truncated, _info = env.step(action + 2)

				### store event in memory bank
				self.add_event(state, previous_state, action, action_log_prob, reward)
				
				### update state
				previous_state = state
				state = new_state
				
				### calculate points scored in game
				if reward > 0:
					points_in_game+=1

				### finish game if done
				if done:
					points_in_all_games.append(points_in_game)
					break
		
		self.compute_reward_history()

		return points_in_all_games
