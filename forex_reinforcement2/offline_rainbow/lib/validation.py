import numpy as np

import torch

from lib import environ
import ptan


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.00, comission=0.0):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }

    val_agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector(), device=device)

    for episode in range(episodes):
        obs = env.reset()
        
        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0
        rand = False
        while True:
            obs_v = [obs]
            out_v,_ = val_agent(obs_v)
            action_idx = out_v[0]
            #action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)
            if rand:
                rand_val = int( np.random.rand(1)[0] *  env.action_space.n)
                action_idx = rand_val
                action = environ.Actions(action_idx)
            if episode_steps > 60 and not env._state.have_position:
                print(env._state.env_name," episode number " , episode, " episode steps " , episode_steps)
                rand = False
                rand_val = 1#int( np.random.rand(1)[0] *  env.action_space.n)
                action_idx = rand_val
                action = environ.Actions(action_idx)



            close_price = env._state._cur_close()

            if action == environ.Actions.Buy and position is None:
                position = close_price
                position_steps = 0
            elif action == environ.Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * comission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)
                position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * comission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                rand = False
                break




        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    return { key: np.mean(vals) for key, vals in stats.items() },env._state.getMeanReward()
