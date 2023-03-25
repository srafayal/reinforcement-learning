import numpy as np
import pandas as pd 
from copy import deepcopy
from gym.spaces.box import Box, Space


import sys
 

import random
import operator
import tensorflow as tf

from keras.layers import Input, Activation, Dense, Flatten, RepeatVector, Reshape

from keras.layers.convolutional import Conv2D
from keras.models import Model
from collections import defaultdict
from IPython.display import clear_output
import itertools

from lib import policies
from blackjack import env
from keras.optimizers import Adam as Adam




class DQNAgent:
    def __init__(self ,n_episodes,batch_size=3,output_dir='model_output/',DDQN=False ):
        self.state_size = 3
        self.action_size =  2
        self.terminal_state = np.array([0,0,0,0])
        # double-ended queue; acts like list, but elements can be added/removed from either end
        # Initialize replay memory D to capacity N
        self.memory = []
        self.gamma = 0.99 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon_start=1.0
        self.epsilon_end=0.1
        self.epsilon_decay_steps=500000
        self.replay_memory_size=50000
        self.replay_memory_init_size=4000
        self.learning_rate = 1e-05 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.update_target_estimator_every=500
        self.q_estimator = self.create_model() # private method 
        self.t_estimator = self.create_model() # private method 
        self.total_t=0
        self.reword_history=[]
        self.epsilon_greedy=   policies.make_epsilon_greedy_policy(self.q_estimator, 2 )
        self.DDQN = DDQN
        self.n_episodes=n_episodes
        self.batch_size=batch_size
        self.output_dir=output_dir
        
    def initialize_state_actions(self ):
        self.default_SA_estimate = new_default
        all_states = self.discretizer.list_all_states()
        all_states.append(self.terminal_state)
        self.state_list = all_states


#     def _build_model(self):
        
          
#         model = Sequential()
#         model.add(Dense(8, input_dim=3, activation='relu')) # 1st hidden layer; states as input
#         model.add(Dense(units = 32, activation = 'relu')) # Adding the second hidden layer
#         model.add(Dense(units = 32, activation = 'relu')) # Adding the third hidden layer
#         model.add(Dense(units = 2)) 
#         model.compile(loss='mean_squared_error',  optimizer=Adam(lr=self.learning_rate))
        
#         #print(model.summary())
#         return model   
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
        #model.compile('rmsprop', 'MSE')
        
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
        
        #print(s)
        X_batch = np.vstack(s)
        y_batch = self.predict(X_batch)
        q_values_next = self.targrt_predict(np.vstack(s2))
        #print(np.amax(q_values_next) )
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
        
    def act(self, X, eps=1.0):

        if np.random.rand() < eps:
            action = self.actions.sample()
            return action

        s =[]
        s.append(  X)  
        Q = self.q_estimator.predict_on_batch(np.array(s))
        action =np.argmax(Q, 1)[0]
        #print(action)
        return action 
    
    
    def learning(self):

        #estimator_copy = ModelParametersCopier(self.q_estimator, self.target_estimator)
        # Get the current time step
        epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)
                 # Populate the replay memory with initial experience
        print("Populating replay memory...")
       

        epsilon = epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        possible_actions = [0,1]
        player_cards, dealer_cards = env.play_init()
        current_policy = self.epsilon_greedy
        epsilon = epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        r, logs=env.play(player_cards, dealer_cards, current_policy,None,epsilon)
        
        done=False
        for i in range(0, len(logs)) :
             
            state, action  = logs[i]
            next_state = state
            reward=0
            # (b) A <- ε-greedy(S,Q)
            # (c) Take action A; observe resultant reward, R, and state, S'
            #next_state, reward, done, details = env.step(action)
            
            #print("s:{}  a:{},  r:{}".format(state,action,reward))
            if logs[i]!=logs[len(logs)-1]:
                next_state,_ = logs[i+1]
            else:
                reward=r
                done= True
                
            self.memory.append((state, action, reward, next_state, done) ) 
            
            
                      
        #return 
        done = False
        i=0
        self.t_estimator.set_weights(self.q_estimator.get_weights()) 
        for e in range(self.n_episodes): # iterate over new episodes of the game
 
                
            player_cards, dealer_cards = env.play_init()
            current_policy = self.epsilon_greedy
            epsilon = epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
            r, logs=env.play(player_cards, dealer_cards, current_policy,None,epsilon)
            
            for i in range(0, len(logs)) :  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
                state, action  = logs[i]
                next_state = state
                reward=0
        
                if logs[i]!=logs[len(logs)-1]:
                    next_state,_ = logs[i+1]
                else:
                    reward=r
                    done= True
                # Maybe update the target estimator
                if self.total_t % self.update_target_estimator_every == 0:
                    self.t_estimator.set_weights(self.q_estimator.get_weights()) 
#                     print("\nCopied model parameters to target network.")
                     
                              
                # If our replay memory is full, pop the first element
                if len(self.memory) == self.replay_memory_size:
                    self.memory.pop(0)

                # Save transition to replay memory
                self.memory.append((state, action, reward, next_state, done) )     # remember the previous timestep's state, actions, reward, etc.        
                state = next_state # set "current state" for upcoming iteration to the current next state        
#                 if done: # episode ends if agent drops pole or we reach timestep 5000
#                     print("episode: {}/{}, score: {}, e: {:.5}" # print the episode's score and agent's epsilon
#                           .format(e, self.n_episodes, reward, epsilon))
#                     break # exit loop
                if len(self.memory) > self.batch_size:
                    X_batch, y_batch = self.create_batch(  self.batch_size)
                    self.train(X_batch, y_batch)
                self.total_t+= 1
                
            if e % 100 == 0:
                print("\rEpisode {}/{}. ...{}".format(e, self.n_episodes,epsilon), end="")
                sys.stdout.flush()
                
            if e % 50 == 0:
                self.save(self.output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

       
    def load(self, name):
        self.q_estimator.load_weights(name)

    def save(self, name):
        self.q_estimator.save_weights(name)