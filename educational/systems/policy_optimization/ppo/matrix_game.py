import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gym.spaces import Discrete
from gym.spaces.box import Box
#import cv2
import jax as jnp

class mat_game(object):
    
    def __init__(self):
        mat0 = np.array([[7,7],[7,7]])
        mat1 = np.array([[4,4],[4,4]])
        mat2 = np.array([[5,5],[5,5]])
        mat3 = np.array([[2,6],[1,8]])
        
        self.mat = np.array([[mat0,mat1],[mat2,mat3]])
        #self.pos = [0,0]
        self.stepNo = 0
        self.n_agents = 2
        self.obs = np.array([[0,0],[0,0]])
        self.pos = self.obs
        self.n_actions = 2
        
        self.observation_space = Box(
                -np.inf,
                np.inf,
                shape=self.obs[0].shape,
                dtype=np.float32#self.obs[0].dtype,
            )
    def step(self,action_list):
        
        if self.stepNo == 0:
            #print(action_list)
            obs =  self.mat[action_list[0],action_list[1]]
            self.stepNo = 1
            reward = 0
            self.pos = obs
            
            return reward,False,obs
        
        else:
            obs = np.array([[0,0],[0,0]])
            self.stepNo = 0
            reward = self.pos[action_list[0],action_list[1]]
            
            return reward,True,obs
        
    def get_avail_agent_actions(self,agent):
        return [1,1]
    
    def get_obs(self):
        return self.pos#np.array([[0,0],[0,0]])

    def reset(self):
        mat0 = np.array([[7,7],[7,7]])
        mat1 = np.array([[4,4],[4,4]])
        mat2 = np.array([[5,5],[5,5]])
        mat3 = np.array([[2,6],[1,8]])
        
        self.mat = np.array([[mat0,mat1],[mat2,mat3]])
        self.pos = np.array([[0,0],[0,0]])
        self.stepNo = 0
        
        return np.array([[0,0],[0,0]]), False

    def get_env_info(self):

        env_info = {}

        env_info["n_actions"] = 2
        env_info["n_agents"] = 2
        env_info["state_shape"] = 4
        env_info["obs_shape"] = 4
        env_info["episode_limit"] = 2

        return env_info
        
        