import numpy as np
import random
import operator
from lib import policies
from blackjack import env
import sys

import pandas as pd 
import pickle
 

class DynaQ():
    def __init__(self, ε=0.1, α = 0.25, planning_steps=3):
        self.name = 'DynaQ'
        self.Q_vals = {}
        self.Model = {}
        self.ε = ε # probability for exploration
        self.α = α # Sarsa step size
        self.x_lookup = {}
        self.y_lookup = {}
        self.Model = np.array([])
        self.planning_steps = planning_steps
        self.state_ix_lookup = {}
        self.reward_table = {}
        self.reword_history=[]
        self.epsilon_greedy=   policies.epsilon_greedy( 2 )
    def initialize_state_actions(self ):
        ''' 
        Initialize Q(s,a) and Model(s,a) for all s∈S and a∈A(s) 
        '''
        all_states = [(p_sum, d_sum, ace) for ace in  (True,False) for p_sum in range(12,23) for d_sum in range(1,23)]
        all_actions = [0,1]

        
        all_sa = [(s, a) for a in all_actions for s in all_states]
        self.x_lookup = {(tuple(state), action): i for i, (state,action) in enumerate(all_sa)}
        self.y_lookup = {tuple(state): i for i, state in enumerate(all_states)}
        self.state_ix_lookup = {self.y_lookup[k]: k for k in self.y_lookup}
        
        
        self.Model = np.zeros((len(self.x_lookup), len(self.y_lookup)))
        self.reward_table = {}
 
        for s in all_states:
            s = tuple(s)
            for a in all_actions:
                self.Q_vals.setdefault(s, {})[a] = 0


#     def choose_randomly(state, actions, period, args={}):
#         return np.random.choice(actions)
    
#     def set_seed(self, seed):
#         self.seed = seed
#         random.seed(seed)
#         np.random.seed(seed)
    
#     def set_policy(self, policy, policy_args={}):
#         self.policy = policy
#         self.policy_args = policy_args
        
        
 
    
    def control(self,  γ=0.1):
        '''
        --Initialize Q(s,a) and Model(s,a) for all s∈S and a∈A(s) 
        Loop forever: 
            (a) S current (nonterminal) state 
            (b) A <- ε-greedy(S,Q) 
            (c) Take action A; observe resultant reward, R, and state, S'
            (d) Q(S,A)<-Q(S,A)+ α[R + γ* max_a Q(S',a)-Q(S,A)] 
            (e) Model(S,A) R,S' (assuming deterministic environment) 
            (f) Loop repeat n times: 
                S<-random previously observed state 
                A<-random action previously taken in S 
                R,S' <- Model(S,A) 
                Q(S,A)<-Q(S,A)+ α[R + γ* max_a Q(S',a)-Q(S,A)]  
        '''  
        possible_actions = [0,1]
        player_cards, dealer_cards = env.play_init()
        current_policy = self.epsilon_greedy
        r, logs=env.play(player_cards, dealer_cards, current_policy,self.Q_vals)

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

            nexts=next_state
            #print(next_state)
            #(d) Q(S,A)<-Q(S,A)+ α[R + γ* max_a Q(S',a)-Q(S,A)] 
            # Q-Learning update
            #next_state = tuple(agent.discretize_space(next_state))
            best_next_action = np.argmax( list(self.Q_vals[next_state].values())) #np.argmax(self.Q_vals[next_state]) 

            target = reward + (γ * self.Q_vals[next_state][best_next_action])
            qUpdate =target - self.Q_vals[state][action]
            
            #print("s:{},  b:{},  a:{},  nb:{} ,r:{}".format(state,self.Q_vals[state][action],action,self.α * qUpdate,target))
            self.Q_vals[state][action]  +=  self.α * qUpdate
            
            # feed the model with experience
            #(e) Model(S,A) R,S' (assuming deterministic environment) 
            self.Model[self.x_lookup[(state, action)], self.y_lookup[next_state]]+= 1
            try:
                self.reward_table[(state, action, next_state)] += self.α * (reward  - self.reward_table[state, action, next_state])
            except KeyError:
                self.reward_table[(state, action, next_state)] = reward
            
            
            for sweeps in range(self.planning_steps):
                #S<-random previously observed state 
                state_ = random.choice(list(self.Q_vals.keys()))
                #A<-random action previously taken in S
                action_ = random.choice(range(2))
                
                #R,S' <- Model(S,A) 
                p_sas = self.Model[self.x_lookup[(state_, action_)], :]
                 
                if np.sum(p_sas) == 0:
                    continue
                
                 
                
                # Choose randomly from the next states based on their probability of being visited but ignoring terminal
                p_sas = p_sas / np.sum(p_sas)
                next_state_ = self.state_ix_lookup[np.random.choice(range(p_sas.size), p=p_sas)]
#                 print(state_)
#                 print(next_state_)
                reward_ = self.reward_table[(state_, action_, next_state_)]
#                 print(reward_)
                
                #Q(S,A)<-Q(S,A)+ α[R + γ* max_a Q(S',a)-Q(S,A)]
                #next_state_ = tuple(agent.discretize_space(next_state_))
                best_next_action_ = np.argmax( list(self.Q_vals[next_state_].values())) #self.Q_vals[tuple(next_state_)]) 
#                 print(self.Q_vals[next_state_])
#                 print(best_next_action_)
#                 print(self.Q_vals[next_state_][best_next_action_])
#                 print('*******************')
                target_ = reward_ + (γ * self.Q_vals[next_state_][best_next_action_])
                qUpdate_ =target_ - self.Q_vals[state_][action_]
#                 print(self.Q_vals[state_][action_])
                self.Q_vals[state_][action_]  +=  self.α * qUpdate_
#                 print(self.α * qUpdate_)
                
            state=nexts
            #print(state)
    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)