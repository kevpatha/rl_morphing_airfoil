import math
import numpy as np
import tensorflow as tf
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import torch as T
import torch.nn as nn
import math
import time
import functools
import serial
import random


def normalize_df(df):
    df = (df-df_mean)/df_std
    return df

def model_average_pred(model0, model1, state):
    
    pred0 = model0(state).numpy()
    pred1 = model1(state).numpy()
    
    return 0.5*pred0[0,0,0]+0.5*pred1[0,0]

class MFC_aileron_Env(gym.Env):

    def __init__(self, start_volt = 50, goal_position=0, steps_per_ep = 200):
 
        self.measure_diff = 0
        self.last_laz = 0
    
        self.flex_to_laz_0 = tf.keras.models.load_model('flex_and_laz_models/LSTM_split_flex_to_laz')
        self.flex_to_laz_1 = tf.keras.models.load_model('flex_and_laz_models/LSTM_flex_to_laz')
        
        # Set up serial communication
        self.ser = serial.Serial()
        self.ser.baudrate = 9600
        self.ser.port = 'COM6'
        self.ser.open()
        time.sleep(2)
        self.new_volt = start_volt
        self.send_Voltage()
        self.laf = np.array((1,10,2))
        
        self.steps_per_ep = steps_per_ep
        self.num_steps = 0
      
        self.act_mean = 50
        self.act_std = 26.8
        self.pos_mean = 760 #755.526 #meausered by flex
        self.pos_std = 50 #76.0765 #measured by flex
        self.laz_mean = 497.5
        self.laz_std = 226
        self.min_position = -8.86
        self.max_position = 2.16
        self.min_volt = -1.87
        self.max_volt = 1.87
        self.goal_position = goal_position # 
        
        self.old_volt = 0
        self.volt = 0
        
        #time checking
        self.delt=0.05
        self.dwait = 0
        self.last_dt = 0
        self.prev_time = time.time()
        self.current_time = time.time()
        
        self.low_state = np.array(
            [self.min_position, self.min_volt, self.min_volt, self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_volt, self.max_volt, self.max_position], dtype=np.float32
        )

        self.action_space = spaces.Discrete(7)
        self.actions = np.array([-6,-4,-2,0,2,4,6])#np.array([-10,-6,-2,0,2,6,10])
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(4,),
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
    
    def normalize_laz(self, laz):
        return (laz-self.laz_mean)/self.laz_std
    
    def send_Voltage(self):
        motor = str(int(self.new_volt)) + '\n'
        self.ser.write(motor.encode())
        
    def get_obs(self, N_avg=1):
    
        buffer_string = ''
        buffer_string = buffer_string + self.ser.read(self.ser.inWaiting()).decode()
        if '\n' in buffer_string:
            lines = buffer_string.split('\n')
            obs = 0
            for ii in range(N_avg):
                obs_str = lines[-2-ii]
                obs += np.array([int(x) for x in obs_str.split(' ')])
                
            obs = obs/N_avg
            return obs

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        self.old_volt = self.state[-1]
        self.new_volt = self.denormalize_volt(self.old_volt) + action
        #self.new_volt = action
        self.volt = self.normalize_volt(self.new_volt)

        if self.volt<= self.min_volt:
            self.volt = self.min_volt
        elif self.volt >= self.max_volt:
            self.volt = self.max_volt

        self.send_Voltage()
        old_and_new_volts = np.array([self.old_volt,self.volt],dtype=float)
  
        self.state[-1] = self.volt
        self.state[1] = self.old_volt
        
        self.dwait += self.delt - self.last_dt
        while True:
            current_time = time.time()
            dt = current_time - self.prev_time 
            if dt >= self.dwait:
                break
        self.last_dt = dt
        self.prev_time = current_time
        
        pos_measurements = self.get_obs()
        laz_obs = pos_measurements[1]
        norm_flex=self.normalize_pos(pos_measurements[0])
        new_lstm_obs = np.array([norm_flex,self.last_laz]).reshape((1,1,2))
        self.laf = np.append(self.laf, new_lstm_obs, axis=1)
        self.laf = np.delete(self.laf, 0, 1)
        
        #new_pos = self.flex_to_laz(self.laf).numpy()
        new_pos = model_average_pred(self.flex_to_laz_0, self.flex_to_laz_1, self.laf)
        #self.last_laz = new_pos[0,0,0]
        self.last_laz = new_pos
        self.state = np.append(self.last_laz, old_and_new_volts)
        
        new_obs = np.append(self.state.reshape(3,), self.goal_position).reshape(-1,1)
        self.obs = np.append(self.obs, new_obs, axis=1)
        self.obs = np.delete(self.obs, 0, 1)
        
        
        
        #error =  self.goal_position - pred_pos
        #reward = -(error*error)
        self.num_steps += 1
        if self.num_steps >= self.steps_per_ep:
            done = True

        return self.obs, self.last_dt, done, laz_obs

    def reset(self, start_volt = 50, goal=0):
        self.goal_position = goal
        self.new_volt = start_volt
        self.send_Voltage()
        self.laf = np.zeros((1,10,2))
        self.obs = np.zeros((4,10))
        time.sleep(2)
        
        self.old_volt = self.normalize_volt(self.new_volt)
        self.volt = self.normalize_volt(self.new_volt)
        pos_measurements = self.get_obs()
        print(pos_measurements)
        old_and_new_volts = np.array([self.old_volt,self.volt],dtype=float)
        norm_flex=self.normalize_pos(pos_measurements[0])
        norm_laz = self.normalize_laz(pos_measurements[1])
        self.last_laz = norm_laz
        time.sleep(0.04)
        
        for i in range(10):
            pos_measurements = self.get_obs()
            norm_flex=self.normalize_pos(pos_measurements[0])
            norm_laz = self.normalize_laz(pos_measurements[1])
            new_laf = [norm_flex, self.last_laz]
            self.laf[0,i,:] = new_laf
            self.last_laz = norm_laz
            time.sleep(0.04)
            self.obs[0,i] = norm_laz
            self.obs[1,i] = self.old_volt
            self.obs[2,i] = self.volt
            self.obs[3,i] = self.goal_position
        
        #new_pos = self.flex_to_laz(self.laf).numpy()
        new_pos = model_average_pred(self.flex_to_laz_0, self.flex_to_laz_1, self.laf)
        
        self.state = np.append(new_pos, old_and_new_volts) 
        self.prev_time = time.time()
        return self.obs, pos_measurements[1]
    
    def next_goal(self, goal):
        self.goal_position = goal
    
    
    
    def end(self):
        self.new_volt = 50
        self.send_Voltage()
        self.ser.close()
    
