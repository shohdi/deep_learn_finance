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

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.0
MAX_GAME_STEPS = 60
STOP_AT_MAX_STEPS = False





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
        self.game_steps_queue = collections.deque(maxlen=100)
        self.max_mean_reward = -100
        self.minLossValue = self.getMinLossValue(1.145)
    
    def getMinLossValue(self,close):
        #assert isinstance(close,float)
        return ((2/( 0.01 * close * 100000))/close) * 100.0

    def getMaxMin(self):
        max = 0.0;
        min = 9999999.0;
        maxAvg = 0.0;
        minAvg = 9999999.0;

        for bar_idx in range(-self.bars_count+1, 1):
            val = self._prices.high[self._offset + bar_idx]
            if(val > 0 and val < min):
                min = val
            if(val > max):
                max = val;
            val = self._prices.low[self._offset + bar_idx]
            if(val > 0 and val < min):
                min = val
            if(val > max):
                max = val;
            
            val = self._prices.avgd[self._offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;
            
            val = self._prices.avgh[self._offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;
            
            val = self._prices.avgm[self._offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;

            val = self._prices.close[self._offset + bar_idx]
            if(val > 0 and val < minAvg):
                minAvg = val
            if(val > maxAvg):
                maxAvg = val;
            
        return min,max,minAvg,maxAvg
            



    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset
        self.minLossValue = self.getMinLossValue(self._cur_close())
        

    @property
    def shape(self):
        if(RETURN_1_D):
            return self.shape1d()
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (9 * self.bars_count + 1 + 1, )
        else:
            return (8*self.bars_count + 1 + 1, )
    
    def shape1d(self):
        if self.volumes:
            return (11, self.bars_count)
        else:
            return (10, self.bars_count)

    def encode1d(self):
        min,max,minAvg,maxAvg = self.getMaxMin()
        deviaAvg = maxAvg - minAvg
        devia = max-min
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = (self._prices.high[self._offset-ofs:self._offset+1] - min)/devia
        res[1] = (self._prices.low[self._offset-ofs:self._offset+1] - min)/devia
        res[2] = (self._prices.open[self._offset-ofs:self._offset+1] - min)/devia
        res[3] = (self._prices.close[self._offset-ofs:self._offset+1] - min)/devia
        res[4] = (self._prices.avgm[self._offset-ofs:self._offset+1] - minAvg)/deviaAvg
        res[5] = (self._prices.avgh[self._offset-ofs:self._offset+1] - minAvg)/deviaAvg
        res[6] = (self._prices.avgd[self._offset-ofs:self._offset+1] - minAvg)/deviaAvg
        res[7] = (self._prices.close[self._offset-ofs:self._offset+1] - minAvg)/deviaAvg
        
        if self.volumes:
            res[8] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 9
        else:
            dst = 8
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self.getTrainReward()
        return res


    def getMeanFromDeque (self,deque):
        sum = 0
        for i in range(len(deque)):
            sum += deque[i]
        
        sum /= len(deque)
        return sum


    def getMeanReward(self):
        mean_reward = self.getMeanFromDeque(self.rewards)
        if(len(self.rewards) > 90 and mean_reward > self.max_mean_reward):
            print(self.env_name," found better mean reward ",self.max_mean_reward," => ",mean_reward)
            self.max_mean_reward = mean_reward 
        return mean_reward



    
    
    def encode(self):
        if(RETURN_1_D):
            return self.encode1d()
        min,max,minAvg,maxAvg = self.getMaxMin()
        deviaAvg = maxAvg - minAvg
        devia = max-min
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            #'high','low','open','close','avgm','avgh','avgd','month','dayofmonth','dayofweek','hour','minute','ask','bid','volume'
            res[shift] = (self._prices.high[self._offset + bar_idx] - min)/devia
            shift += 1
            res[shift] = (self._prices.low[self._offset + bar_idx] - min)/devia
            shift += 1
            res[shift] = (self._prices.open[self._offset + bar_idx] - min)/devia
            shift += 1
            res[shift] = (self._prices.close[self._offset + bar_idx] - min)/devia
            shift += 1
            res[shift] = (self._prices.avgm[self._offset + bar_idx] - minAvg)/deviaAvg
            shift += 1
            res[shift] = (self._prices.avgh[self._offset + bar_idx] - minAvg)/deviaAvg
            shift += 1
            res[shift] = (self._prices.avgd[self._offset + bar_idx] - minAvg)/deviaAvg
            shift += 1
            res[shift] = (self._prices.close[self._offset + bar_idx] - minAvg)/deviaAvg
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
        self.minLossValue = self.getMinLossValue(self._cur_close())
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = self._cur_ashtry();
            reward -= self.commission_perc
            self.game_steps = 0
        elif (action == Actions.Close or self.getTrainReward() <= (-1 * self.minLossValue) or self.getTrainReward() >= (2 * self.minLossValue) ) and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += self.getTrainReward()
            else:
                reward -= 0.05; #spread
            self.game_done+=1
            self.rewards.append(self.getReward())
            self.writer.add_scalar("shohdi-"+self.env_name+"-reward",self.getMeanReward(),self.game_done)
            self.game_steps_queue.append(self.game_steps);
            self.writer.add_scalar("shohdi-"+self.env_name+"-steps",self.getMeanFromDeque(self.game_steps_queue),self.game_done)
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