# %% [markdown]
# # Quantile Regression Q-Learning

# %% [markdown]
# ## Imports

# %%
import sys
sys.path.insert(0, "/home/aditya-ws/workspace/personal/thirdparty/baselines")
import gym
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output
from matplotlib import pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
import math
from tqdm import trange

from utils.wrappers import *
from agents.DQN import Model as DQN_Agent
from utils.ReplayMemory import ExperienceReplayMemory
from networks.layers import NoisyLinear

from utils.hyperparameters import Config
from morl import api, experiences as exp, external_utils as extu

# %% [markdown]
# ## Hyperparameters

# %%
config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 60000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA=0.99
config.LR=1e-4

#memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 100000
config.BATCH_SIZE = 32

#Learning control variables
config.LEARN_START = 10000
config.MAX_FRAMES=1000000

#Nstep controls
config.N_STEPS=1

#Quantile Regression Parameters
config.QUANTILES=51

# %% [markdown]
# ## Network

# %%
class QRDQN(nn.Module):
    def __init__(self, input_shape, num_actions, quantiles=51):
        super(QRDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.quantiles=quantiles

        self.head = torch.nn.Sequential(
            extu.AsDtype(torch.float32),
            extu.PixelNormalize()
        )

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions*self.quantiles)

    def forward(self, x):
        if isinstance(env, api.MorlEnv):
            x = self.head(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1, self.num_actions, self.quantiles)

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

# %% [markdown]
# ## Agent

# %%
class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None):
        self.num_quantiles = config.QUANTILES
        self.cumulative_density = torch.tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles), device=config.device, dtype=torch.float)
        self.quantile_weight = 1.0 / self.num_quantiles

        super(Model, self).__init__(static_policy, env, config)

    def declare_networks(self):
        self.model = QRDQN(self.num_feats, self.num_actions, quantiles=self.num_quantiles)
        self.target_model = QRDQN(self.num_feats, self.num_actions, quantiles=self.num_quantiles)

    def next_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.num_quantiles), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                quantiles_next[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action).squeeze(dim=1)

            quantiles_next = batch_reward + (self.gamma*quantiles_next)

        return quantiles_next

    def morl_next_distribution(self, tensor_exp: exp.TensorExperience) -> torch.Tensor:
        max_next_action = self.get_max_next_state_action(tensor_exp.next_states)
        quantiles_next = self.target_model(tensor_exp.next_states).gather(1, max_next_action).squeeze(dim=1)
        return tensor_exp.rewards + (self.gamma * quantiles_next * (~tensor_exp.dones).float())

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.num_quantiles)

        quantiles = self.model(batch_state)
        quantiles = quantiles.gather(1, batch_action).squeeze(1)

        quantiles_next = self.next_distribution(batch_vars)

        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0,1)
        loss = loss.mean(1).sum(-1).mean()

        return loss

    def morl_compute_loss(self, tensor_exp: exp.TensorExperience) -> torch.Tensor:
        batch_action = tensor_exp.actions.long()
        batch_state = tensor_exp.states
        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.num_quantiles)

        quantiles = self.model(batch_state)
        quantiles = quantiles.gather(1, batch_action).squeeze(1)

        # quantiles_next = self.next_distribution(batch_vars)
        quantiles_next = self.morl_next_distribution(tensor_exp)

        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0,1)
        loss = loss.mean(1).sum(-1).mean()

        return loss


    def get_action(self, s, eps):
        with torch.no_grad():
            if np.random.random() < eps:
                result = np.random.randint(0, self.num_actions)
                if isinstance(env, api.MorlEnv):
                    result = np.asarray([result]).reshape(1, 1)
                return result
            if isinstance(env, api.MorlEnv):
                X = torch.as_tensor(s, device=self.device)
            else:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
            a = (self.model(X) * self.quantile_weight).sum(dim=2).max(dim=1)[1]
            if isinstance(env, api.MorlEnv):
                return a.view(-1, 1).cpu().numpy()
            return a.item()

    def get_max_next_state_action(self, next_states):
        next_dist = self.target_model(next_states)*self.quantile_weight
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_quantiles)

# %% [markdown]
# ## Plot Results

# %%
def plot(frame_idx, rewards, losses, sigma, elapsed_time):
    print(f'frame {frame_idx}. reward: {np.mean(rewards[-10:])}. time: {elapsed_time}')
    # clear_output(True)
    # plt.figure(figsize=(20,5))
    # plt.subplot(131)
    # plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
    # plt.plot(rewards)
    # if losses:
    #     plt.subplot(132)
    #     plt.title('loss')
    #     plt.plot(losses)
    # if sigma:
    #     plt.subplot(133)
    #     plt.title('noisy param magnitude')
    #     plt.plot(sigma)
    # plt.show()

# %% [markdown]
# ## Training Loop

# %%
from morl import environments as envs
from morl import core
core.set_default_torch_device("cuda")
start=timer()

env_id = "PongNoFrameskip-v4"
# env    = make_atari(env_id)
# env    = wrap_deepmind(env, frame_stack=False)
# env    = wrap_pytorch(env)
env = envs.EnvPool.make_ale_env(
    game="Pong",
    n_envs=1,
    base_seed=0,
    frame_stack=1,
)
model = Model(env=env, config=config)

episode_reward = 0

if not isinstance(env, api.MorlEnv):
    observation = env.reset()
for frame_idx in trange(1, config.MAX_FRAMES + 1):
    if isinstance(env, api.MorlEnv):
        observation = env.get_states_()
    epsilon = config.epsilon_by_frame(frame_idx)

    action = model.get_action(observation, epsilon)
    prev_observation=observation
    if isinstance(env, api.MorlEnv):
        exp_ = env.step_(action)
        observation, reward, done = exp_.next_states[0], exp_.rewards.item(), exp_.dones.item()
    else:
        observation, reward, done, _ = env.step(action)
    observation = None if done else observation

    if isinstance(env, api.MorlEnv):
        model.morl_update(exp_, frame_idx)
    else:
        model.update(prev_observation, action, reward, observation, frame_idx)
    episode_reward += reward

    if done:
        if not isinstance(env, api.MorlEnv):
            model.finish_nstep()
        model.reset_hx()
        if not isinstance(env, api.MorlEnv):
            observation = env.reset()
        model.save_reward(episode_reward)
        episode_reward = 0

        if np.mean(model.rewards[-10:]) > 19:
            plot(frame_idx, model.rewards, None, None, timedelta(seconds=int(timer()-start)))
            break

    if frame_idx % 10000 == 0:
        plot(frame_idx, model.rewards, None, None, timedelta(seconds=int(timer()-start)))

model.save_w()
env.close()

# %%
