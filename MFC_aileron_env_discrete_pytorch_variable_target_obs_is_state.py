import math

import numpy as np
import tensorflow as tf
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding


# Load state data
col_names = ['Flex', 'Laz', 'st_volt', 'end_volt', 'time_step', 'tot_time']#, 'time_step', 'total_time']

data_filepath0 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed0.csv'
data_filepath1 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed1.csv'
data_filepath2 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed2.csv'
data_filepath3 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed3.csv'
data_filepath4 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed4.csv'
data_filepath5 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed5.csv'
data_filepath6 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed6.csv'
data_filepath7 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed7.csv'
data_filepath8 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed8.csv'
data_filepath9 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed9.csv'
data_filepath10 = 'data/MFC_dynamics_data0_step2_range31_shuffle_seed10.csv'

raw_data0 = pd.read_csv(data_filepath0, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data1 = pd.read_csv(data_filepath1, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data2 = pd.read_csv(data_filepath2, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data3 = pd.read_csv(data_filepath3, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data4 = pd.read_csv(data_filepath4, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data5 = pd.read_csv(data_filepath5, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data6 = pd.read_csv(data_filepath6, names=col_names,
                        na_values = '?', comment='\t', sep=',')
#raw_data7 = pd.read_csv(data_filepath7, names=col_names,
 #                       na_values = '?', comment='\t', sep=',')
raw_data8 = pd.read_csv(data_filepath8, names=col_names,
                        na_values = '?', comment='\t', sep=',')
#test dataset
raw_data9 = pd.read_csv(data_filepath9, names=col_names,
                        na_values = '?', comment='\t', sep=',')
raw_data10 = pd.read_csv(data_filepath10, names=col_names,
                         na_values = '?', comment='\t', sep=',')



raw_data = []
raw_data.append(raw_data0)
raw_data.append(raw_data1)
raw_data.append(raw_data2)
raw_data.append(raw_data3)
raw_data.append(raw_data4)
raw_data.append(raw_data5)
raw_data.append(raw_data6)
#raw_data.append(raw_data7)
raw_data.append(raw_data8)
raw_data.append(raw_data9)

raw_data = pd.concat(raw_data, axis=0, ignore_index=True)
dataset = raw_data.copy()
#dataset = dataset[1::2]
#Clean Data from outlires
dataset = dataset[dataset['Flex']<1023]  # might decide not to do this later
dataset = dataset[dataset['Laz']>0]      #might decide not to do this later

dataset.drop('time_step', inplace=True, axis=1)
dataset.drop('Flex', inplace=True, axis=1)    #drop flwx when using laz for input
#dataset.drop('Laz', inplace=True, axis=1)     #drop Laz when using Flex for input
run_time = dataset.pop('tot_time')
df = dataset #dataset[5::6]
df.tail()

df_mean = df.mean()
df_std = df.std()

test_ds = raw_data10.copy()
test_ds = test_ds[test_ds['Flex']<1023]  # might decide not to do this later
test_ds = test_ds[test_ds['Laz']>0]      #might decide not to do this later

test_ds.drop('time_step', inplace=True, axis=1)
test_ds.drop('Flex', inplace=True, axis=1)   #drop flex when using laz as input
#test_ds.drop('Laz', inplace=True, axis=1)     #drop laz when using flex as input
test_run_time = test_ds.pop('tot_time')
test_df = test_ds[-20000:] #dataset[5::6]

def normalize_df(df):
    df = (df-df_mean)/df_std
    return df

df = normalize_df(df).values
test_df = normalize_df(test_df).values

class MFC_aileron_Env(gym.Env):

    def __init__(self, goal_position=0, steps_per_ep = 200):
        # LOAD MODELS
        self.LSTM_laz = tf.keras.models.load_model('env_models/LSTM_laz_dynamics')
        
        self.steps_per_ep = steps_per_ep
        self.num_steps = 0
        self.df = df
        self.df_length = len(self.df)
        
        self.test_df = test_df

        self.act_mean = 50
        self.act_std = 26.8
        self.pos_mean = 497.5
        self.pos_std = 226
        self.min_position = -2.21
        self.max_position = 2.33
        self.min_volt = -1.87
        self.max_volt = 1.87
        self.goal_position = goal_position 
        
        self.min_action = 0
        self.max_action = 100
        
        self.low_state = (np.ones((10,4))* np.array(
            [self.min_position, self.min_volt, self.min_volt, self.min_position], dtype=np.float32
        )).T
        self.high_state = (np.ones((10,4))*np.array(
            [self.max_position, self.max_volt, self.max_volt, self.max_position], dtype=np.float32
        )).T

        self.action_space = spaces.Discrete(7)
        self.actions = np.array([-6, -4, -2, 0, 2, 4, 6])
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(4,10),
            dtype=np.float32
        )


        self.seed()
        self.reset()

    def normalize_volt(self, act):
        norm_act = (act-self.act_mean)/self.act_std
        return norm_act
    def denormalize_volt(self, norm_act):
        act = norm_act*self.act_std+self.act_mean
        return act
    
    def normalize_action(self, act):
        norm_act = act/20
        return norm_act
    
    def denormalize_action(self, norm_act):
        act = norm_act*20
        return act

    def normalize_pos(self, pos):
        norm_pos = (pos-self.pos_mean)/self.pos_std
        return norm_pos

    def denormalize_pos(self, norm_pos):
        pos = norm_pos*self.pos_std+self.pos_mean
        return pos

    def model_average_pred(self):
        pred2 = self.LSTM_laz(self.state).numpy()
        return pred2 
   

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        old_volt = self.state[-1, -1, -1]
        new_volt = self.denormalize_volt(old_volt) + self.actions[action]
        volt = self.normalize_volt(new_volt)

        if volt<= self.min_volt:
            volt = self.min_volt
        elif volt >= self.max_volt:
            volt = self.max_volt

        self.state[-1,-1,-1] = volt
        self.state[-1,-1,1] = old_volt
        pred_pos = self.model_average_pred()
        new_state = np.append(pred_pos, np.array([old_volt,volt],
                                                     dtype=float)).reshape((1,1,3))
        self.state = np.append(self.state, new_state, axis=1)
        self.state = np.delete(self.state, 0,1)
        new_obs = np.append(new_state.reshape(3,), self.goal_position).reshape((4,1))
        self.obs = np.append(self.obs, new_obs, axis=1)
        self.obs = np.delete(self.obs, 0, 1)
        
        error =  self.goal_position - pred_pos
        reward = -(error*error)
        self.num_steps += 1
        if self.num_steps >= self.steps_per_ep:
            done = True

        return self.obs, reward, done, {}

    def reset(self, goal=0):
        start_ind = np.random.randint(self.df_length-100)
        self.state = self.df[start_ind:start_ind+10,:].reshape((1,10,3))
        self.num_steps = 0
        self.goal_position = goal
        goal_vec = np.ones((1,10,1))*goal
        self.obs = np.append(self.state, goal_vec, axis = 2)
        self.obs = self.obs.reshape((10,4)).T
        return self.obs
    
    def next_goal(self, goal):
        self.goal_position = goal
    
    def test_reset(self, goal=0):
        start_ind = 6500
        self.goal_position = goal
        self.state = self.test_df[start_ind:start_ind+10,:].reshape((1,10,3))
        goal_vec = np.ones((1,10,1))*goal
        self.obs = np.append(self.state, goal_vec, axis = 2)
        self.obs = self.obs.reshape((10,4)).T
        self.num_steps = 0
        return self.obs
