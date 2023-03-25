from copy import deepcopy
# from gym.spaces.box import Box, Space

import random
import operator
import tensorflow as tf

from keras.layers import Input, Activation, Dense, Flatten, RepeatVector, Reshape

from keras.layers.convolutional import Conv2D
from keras.models import Model
from collections import defaultdict
from IPython.display import clear_output
import itertools

import numpy as np
 
import pandas as pd 
from keras.optimizers import Adam as Adam
import sys
class DQNAgent:
    def __init__(self, state_size, action_size, env ,decode_states, n_episodes,batch_size=3,output_dir='model_output/', DDQN=False ):
#         self.state_size = 4
#         self.action_size =  6
        self.state_size =state_size
        self.action_size =action_size
        # double-ended queue; acts like list, but elements can be added/removed from either end
        # Initialize replay memory D to capacity N
        self.memory = []
        self.gamma = 0.90 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon_start=1.0
        self.epsilon_end=0.1
        self.epsilon_decay_steps=500000
        self.replay_memory_size=10000
        self.replay_memory_init_size=400
        self.learning_rate = .001 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.update_target_estimator_every=1000
        self.q_estimator = self.create_model() # private method 
        self.t_estimator = self.create_model() # private method 
        self.total_t=0
        self.reword_history=[]
        self.DDQN = DDQN
        self.env = env        
        self.n_episodes=n_episodes
        self.batch_size=batch_size
        self.output_dir=output_dir
        self.decode_states=decode_states
        
    def initialize_state_actions(self, new_default=0, do_nothing_action=None, do_nothing_bonus=1):
  
        all_states1 = [ s  for s in range(self.state_size)]
 


    def create_model(self, hidden_dims=[64, 64]):
        X = Input(shape=(self.state_size, ))
        net = RepeatVector(self.state_size)(X)
        net = Reshape([self.state_size, self.state_size, 1])(net)
        for h_dim in hidden_dims:
            net = Conv2D(h_dim, [3, 3], padding='SAME')(net)
            net = Activation('relu')(net)
        net = Flatten()(net)
        net = Dense(self.action_size)(net)



        model = Model(inputs=X, outputs=net)
        model.compile(loss='mean_squared_error',  optimizer=Adam(lr=self.learning_rate))
#         model.compile(loss='mean_squared_error',  optimizer=Adam(lr=self.learning_rate))
        
        #print(model.summary())
        return model   
 
    def train(self, X_batch, y_batch):
        return self.q_estimator.train_on_batch(X_batch, y_batch)
    
    def predict(self, X_batch):
        return self.q_estimator.predict_on_batch(X_batch)
    def targrt_predict(self, X_batch):
        return self.t_estimator.predict_on_batch(X_batch)
    
    def create_batch(self,  batch_size):
         
        sample = random.sample( self.memory, batch_size)
        sample = np.asarray(sample)
        s = sample[:, 0]
        a = sample[:, 1].astype(np.int8)
        r = sample[:, 2]
        s2 = sample[:, 3]
        d = sample[:, 4] * 1.
        

        X_batch = np.vstack(s)
        y_batch = self.predict(X_batch)
        q_values_next = self.targrt_predict(np.vstack(s2))
        
        if self.DDQN==False :
            y_batch[np.arange(batch_size), a] = r + (1 - d) * self.gamma * np.amax(q_values_next, axis=1)  
        else:
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = self.targrt_predict(np.vstack(s2))
            y_batch[np.arange(batch_size), a] = r + (1 - d) *  self.gamma * q_values_next_target[np.arange(batch_size), best_actions]
    
        return X_batch, y_batch

    def print_info(episode, reward, eps):

        msg = f"[Episode {episode:>5}] Reward: {reward:>5} EPS: {eps:>3.2f}"

        print(msg)

    
#     def act(self, X, eps=1.0):

#         if np.random.rand() < eps:
#             action =  np.random.choice(self.action_size)
#             return action

#         s =[]
#         s.append(  X)  
#         Q = self.q_estimator.predict_on_batch(np.array(s))
#         action =np.argmax(Q, 1)[0]
#         #print(action)
#         return action  
    
    def learning(self):

        #estimator_copy = ModelParametersCopier(self.q_estimator, self.target_estimator)
        # Get the current time step
        epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)
                 # Populate the replay memory with initial experience

        state = self.env.reset()
        
        
        for i in range(self.replay_memory_init_size):
 
            epsilon = epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
            action = self.act(state,epsilon) 
 
            next_state, reward, done, details = self.env.step(action)
 
            self.memory.append(( self.decode_states[state] , action, reward,  self.decode_states[next_state] , done)) 
            if done:
                state = self.env.reset()
            else:
                state = next_state
                      
        #return 
        done = False
        i=0
        self.t_estimator.set_weights(self.q_estimator.get_weights()) 
        for e in range(self.n_episodes): # iterate over new episodes of the game
            if e % 1 == 0:
                print("\rEpisode {}/{}. ...{}".format(e, self.n_episodes,epsilon), end="")
                sys.stdout.flush()
                
            state = self.env.reset() # reset state at start of each new episode of the game
 
            for time in itertools.count():  
        #         env.render()
                # Maybe update the target estimator
                if self.total_t % self.update_target_estimator_every == 0:
                    self.t_estimator.set_weights(self.q_estimator.get_weights()) 
#
                     
                epsilon = epsilons[min(self.total_t, self.epsilon_decay_steps-1)]    
                action = self.act(state,epsilon) 
                next_state, reward, done, _ = self.env.step(action)             
 
              
                
                # If our replay memory is full, pop the first element
                if len(self.memory) == self.replay_memory_size:
                    self.memory.pop(0)
                
                # Save transition to replay memory
                self.memory.append(( self.decode_states[state] , action, reward,  self.decode_states[next_state] , done))  # remember the previous timestep's state, actions, reward, etc.        
                state = next_state # set "current state" for upcoming iteration to the current next state        
                if done: # episode ends if agent drops pole or we reach timestep 5000
                    break # exit loop
 
                self.total_t+= 1

                if len(self.memory) > self.batch_size:
                    X_batch, y_batch = self.create_batch( self.batch_size)
                    self.train(X_batch, y_batch)
                

                 
            
                
#             if e % 50 == 0:
#                 self.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
                
                
 
 
    
    def act(self, state,epsilon):
        if np.random.rand() <= epsilon: # if acting randomly, take random action
            action =  np.random.choice(self.action_size)
            return action
        s =[]
        s.append(  self.decode_states[state])  
        act_values = self.q_estimator.predict(np.array(s)) # if not acting randomly, predict reward value based on current state

        return np.argmax(act_values[0]) # pick the action that will give the highest reward 
    

        
    def load(self, name):
        self.q_estimator.load_weights(name)

    def save(self, name):
        self.q_estimator.save_weights(name)