import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import collections
from tensorboardX import SummaryWriter
import math
import ptan


from . import data

DEFAULT_BARS_COUNT = 30
DEFAULT_COMMISSION_PERC = 0.0
MAX_GAME_STEPS = 30

RETURN_1_D = False

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2



class State15:
    def __init__(self,env_name,writer, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=False):
        assert isinstance(env_name,str)
        assert not (env_name == None or env_name == '')
        assert isinstance(writer,SummaryWriter)
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.env_name = env_name
        self.writer = writer
        self.game_done = 0
        self.rewards = collections.deque(maxlen=100)
        self.game_steps = 0
        
    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        if(RETURN_1_D):
            return self.shape1d()
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (8 * self.bars_count + 1 + 1, )
        else:
            return (7*self.bars_count + 1 + 1, )
    
    def shape1d(self):
        if self.volumes:
            return (10, self.bars_count)
        else:
            return (9, self.bars_count)

    def encode1d(self):
        current_close = self._offset_close()
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = (self._prices.high[self._offset-ofs:self._offset+1] - current_close)/current_close
        res[1] = (self._prices.low[self._offset-ofs:self._offset+1] - current_close)/current_close
        res[2] = (self._prices.open[self._offset-ofs:self._offset+1] - current_close)/current_close
        res[3] = (self._prices.close[self._offset-ofs:self._offset+1] - current_close)/current_close
        res[4] = (self._prices.avgm[self._offset-ofs:self._offset+1] - current_close)/current_close
        res[5] = (self._prices.avgh[self._offset-ofs:self._offset+1] - current_close)/current_close
        res[6] = (self._prices.avgd[self._offset-ofs:self._offset+1] - current_close)/current_close
        
        if self.volumes:
            res[7] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 8
        else:
            dst = 7
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self.getTrainReward()
        return res


    def getMeanReward(self):
        sum = 0
        for i in range(len(self.rewards)):
            sum += self.rewards[i]
        
        sum /= len(self.rewards)
        return sum

    def normCustomArray(self,arrIn):
        current_close = self._offset_close()
        num = 7;
        if(self.volumes):
            num = 8;
        
  
        for i in range(self.bars_count):
            

            for j in range(7):
                my_index = (i*num)+j;
                arrIn[my_index] = (arrIn[my_index] - current_close)/current_close
               
        return arrIn;

    
    
    def encode(self):
        if(RETURN_1_D):
            return self.encode1d()
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            #'high','low','open','close','avgm','avgh','avgd','month','dayofmonth','dayofweek','hour','minute','ask','bid','volume'
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.open[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.avgm[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.avgh[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.avgd[self._offset + bar_idx]
            shift += 1
            #res[shift] = self._prices.month[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.dayofmonth[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.dayofweek[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.hour[self._offset + bar_idx]
            #shift += 1
            #res[shift] = self._prices.minute[self._offset + bar_idx]
            #shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1            
        res[shift] = float(self.have_position)
        shift += 1
        res[shift] = self.getTrainReward();
        
        res = self.normCustomArray(res);
        return res

    def getTrainReward(self):
        if not self.have_position:
            return 0.0
        return ((self._cur_exit_pos() - self.open_price)/self.open_price)*100
    def getReward(self):
        if not self.have_position:
            return 0.0
        else:
            return ((self._cur_exit_pos() - self.open_price) * (0.01 * 100000))/10.0;
        
    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        if not self.have_position:
            return self._cur_ashtry();
        else:
            return self._cur_exit_pos();

    def _cur_ashtry(self):
        
        return self._prices.ask[self._offset];
    def _cur_exit_pos(self):
       
        return self._prices.bid[self._offset];
        
    def _offset_close(self):
        return self._prices.close[self._offset];

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = self._cur_ashtry();
            reward -= self.commission_perc
            self.game_steps = 0
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += self.getTrainReward()
            else:
                reward -= 0.05; #spread
            self.game_done+=1
            self.rewards.append(self.getReward())
            self.writer.add_scalar("shohdi-"+self.env_name+"-reward",self.getMeanReward(),self.game_done)
            self.game_steps = 0
            self.have_position = False
            self.open_price = 0.0
        elif self.getReward() <= -0.5 and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += self.getTrainReward();
            else:
                reward -= 0.05; #spread
            self.game_done+=1
            self.rewards.append(self.getReward())
            self.writer.add_scalar("shohdi-"+self.env_name+"-reward",self.getMeanReward(),self.game_done)
            self.game_steps = 0
            self.have_position = False
            self.open_price = 0.0
        elif self.game_steps >= MAX_GAME_STEPS and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += self.getTrainReward();
            else:
                reward -= 0.05; #spread
            self.game_done+=1
            self.rewards.append(self.getReward())
            self.writer.add_scalar("shohdi-"+self.env_name+"-reward",self.getMeanReward(),self.game_done)
            self.game_steps = 0
            self.have_position = False
            self.open_price = 0.0
        if(self.have_position):
            self.game_steps +=1
        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += ((close - prev_close) / prev_close) * 100;

        return reward, done



class State:
    def __init__(self,env_name,writer, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(env_name,str)
        assert not (env_name == None or env_name == '')
        assert isinstance(writer,SummaryWriter)
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.env_name = env_name
        self.writer = writer
        self.game_done = 0
        self.rewards = collections.deque(maxlen=100)
        
    def getMeanReward(self):
        sum = 0
        for i in range(len(self.rewards)):
            sum += self.rewards[i]
        
        sum /= len(self.rewards)
        return sum


    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3*self.bars_count + 1 + 1, )

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1            
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res


    def getReward(self):
        if not self.have_position:
            return 0.0
        else:
            return ((self._cur_exit_pos() - self.open_price) * (0.01 * 100000))/10.0;
        


    def _cur_ashtry(self):
        
        return self._prices.ask[self._offset];
    def _cur_exit_pos(self):
       
        return self._prices.bid[self._offset];

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        rel_close = self._cur_ashtry();
        if not self.have_position:
            rel_close= self._cur_ashtry();
        else:
            rel_close= self._cur_exit_pos();
        #open = self._prices.open[self._offset]
        return rel_close


    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.game_done+=1
            self.rewards.append(self.getReward())
            self.writer.add_scalar("shohdi-"+self.env_name+"-reward",self.getMeanReward(),self.game_done)
            self.have_position = False
            self.open_price = 0.0
        elif self.getReward() <= -0.5 and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.game_done+=1
            self.rewards.append(self.getReward())
            self.writer.add_scalar("shohdi-"+self.env_name+"-reward",self.getMeanReward(),self.game_done)
            self.have_position = False
            self.open_price = 0.0
            

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,env_name,writer, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True,state_15 = True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=True, volumes=False):
        assert isinstance(env_name,str)
        assert not (env_name == None or env_name == '')
        assert isinstance(writer,SummaryWriter)
        assert isinstance(prices, dict)
        self.reward_on_close = reward_on_close
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        elif state_15:
            self._state = State15(env_name,writer,bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(env_name,writer,bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.action_space.sample = (lambda : self.mySample())
        print("action space sample ",self.action_space.sample())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()
        self._step = 0

    def mySample(self):
        ret = int( math.floor( (np.random.random() * 30)))
        if(ret > 2):
            ret = 0
        return ret


    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        self._step +=1
        if(self._step == 1):
            print("obs ",obs)
            print("reward ",reward)
            print("reward on close ",self.reward_on_close)
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)



class ShohdiEpsilonGreedyActionSelector(ptan.actions.EpsilonGreedyActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        super(ShohdiEpsilonGreedyActionSelector, self).__init__(epsilon,selector)
        

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size,n_actions = scores.shape
        first_act = n_actions
        n_actions = n_actions * 5
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        rand_actions = np.array(list(map(lambda v : v if v < first_act else 0  ,rand_actions)),dtype=np.int)
        actions[mask] = rand_actions
        return actions