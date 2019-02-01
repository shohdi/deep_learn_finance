#!/usr/bin/env python3
import os
import gym
from gym import wrappers

import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common,environ, data, validation


DEFAULT_STOCKS = "data/year_1.csv"
DEFAULT_VAL_STOCKS = "data/year_2.csv"
#DEFAULT_STOCKS = "/home/shohdi/projects/deep_learn_finance/forex_reinforcement2/test_book_env/data/year_1.csv"
#DEFAULT_VAL_STOCKS = "/home/shohdi/projects/deep_learn_finance/forex_reinforcement2/test_book_env/data/year_2.csv"
STATE_15 = True
BARS_COUNT = 10
CHECKPOINT_EVERY_STEP = 1000000
VALIDATION_EVERY_STEP = 100000

# n-step
REWARD_STEPS = 2

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()
        '''
        self.conv = nn.Sequential(
            nn.Conv1d(input_shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5),
            nn.ReLU()
        )
        '''

        self.conv = nn.Sequential(
            nn.Linear(input_shape[0], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    

    # next state distribution
    # dueling arch -- actions from main net, distr from tgt_net

    # calc at once both next and cur states
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()

def calculateModelParams(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


if __name__ == "__main__":
    params = common.HYPERPARAMS['shohdi']
    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, action="store_true", help="data file")
    parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, action="store_true", help="validation data file")
    parser.add_argument("-r", "--run", default="shohdi-forex", help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)

    writer = SummaryWriter(comment="-" + args.run + "-rainbow")


    stock_data = {"YANDEX": data.load_relative(args.data)}
    env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)
    env_tst = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)

    val_data = {"YANDEX": data.load_relative(args.valdata)}
    env_val = environ.StocksEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)

    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    calculateModelParams(net)
    tgt_net = ptan.agent.TargetNet(net)
    EPSILON_START = params["epsilon_start"]
    EPSILON_STEPS = params["epsilon_frames"]
    EPSILON_STOP =  params["epsilon_final"]
    selector = environ.ShohdiEpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START

    with common.RewardTracker(writer, params['stop_reward'],group_rewards=100) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - frame_idx / EPSILON_STEPS)
            

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = calc_loss(batch, net, tgt_net.target_model,
                                               params['gamma'] ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()
            

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            
            
            
            if frame_idx % CHECKPOINT_EVERY_STEP == 0:
                idx = frame_idx // CHECKPOINT_EVERY_STEP
                torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))

            if frame_idx % VALIDATION_EVERY_STEP == 0:
                res = validation.validation_run(env_tst, net, device=device,epsilon=0.0)
                for key, val in res.items():
                    writer.add_scalar(key + "_test", val, frame_idx)
                res = validation.validation_run(env_val, net, device=device,epsilon=0.0)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, frame_idx)
            
