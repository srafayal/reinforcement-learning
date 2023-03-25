import numpy as np
import random
import operator
from lib import policies
from blackjack import env
import sys

import pandas as pd 
import pickle

class QLearning():
    def __init__(self):
        self.name = 'QLearning'
 
        self.Q_vals = {}
        self.epsilon_greedy=   policies.epsilon_greedy( 2 )
        # Hyperparameters

        self.alpha = 0.5 # Sarsa step size
        self.gamma= .9
        self.terminal_state = np.array([0,0,True])
        self.terminal_state1 = np.array([0,0,False])
        
    def initialize_state_actions(self ):
        ''' 
        Algorithm parameters: step size ↵2(0,1], small ">0 
        Initialize Q(s,a), for all s2S+,a2A(s), arbitrarily except that Q(terminal,·)=
        '''
        all_states = [(p_sum, d_sum, ace)  for p_sum in range(12,23) for d_sum in range(1,23) for ace in  (True,False)]
        
        all_actions = [0,1]
        all_states.append(self.terminal_state)
        all_states.append(self.terminal_state1)
        #all_sa = [(s, a) for a in all_actions for s in all_states]
 
        for s in all_states:
            s = tuple(s)
            for a in all_actions:
                self.Q_vals.setdefault(s, {})[a] = 0.0

 
 
 
    
    def control(self,epsilon):
        '''
        Initialize S 
        Choose A from S using policy derived from Q (e.g., ε-greedy) 
        Loop for each step of episode: 
            Choose A from S using policy derived from Q (e.g., ε-greedy) 
            Take action A, observe R, S'
            Q(S,A)<-Q(S,A)+ α[R + γ*max_a Q(S',a)-Q(S,A)]
            S<-S'
        until S is terminal
        '''  
        
        possible_actions = [0,1]
        player_cards, dealer_cards = env.play_init()
        current_policy = self.epsilon_greedy
        r, logs=env.play(player_cards, dealer_cards, current_policy,self.Q_vals,epsilon)
#         print(logs)
#         print(r)
#         logs.append([( 0.0,  0.0, state[2]) , 0]) 
        for i in range(0, len(logs)) :
            
            state, action  = logs[i]
            next_state = state
            reward=0
            if logs[i]!=logs[len(logs)-1]:
                next_state,_ = logs[i+1]
            else:
                reward=r
            # (b) A <- ε-greedy(S,Q)
            # (c) Take action A; observe resultant reward, R, and state, S'
#             next_state,_ = logs[i+1]
#             if logs[i]==logs[len(logs)-2]: 
#                 reward=r
#             print( state)     
#             print(reward)
            # TD Update
             
            best_next_action = np.argmax(self.Q_vals[next_state])
#             print(best_next_action)
             
 
            self.Q_vals[state][action]   =self.Q_vals[state][action]  +  self.alpha * ( reward+ self.gamma *self.Q_vals[next_state][best_next_action]  -self.Q_vals[state][action]  )  
            state=next_state
        
    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)