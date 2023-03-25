import numpy as np
import random
import operator
import pandas as pd 
import pickle

class QLearning():
    def __init__(self, state_size, action_size, env):
        self.name = 'Sarsa'
        
        self.Q_vals = {}
        
        # Hyperparameters
        self.ε = 0.1 # probability for exploration
        self.α = 0.1 # Sarsa step size
        self.γ= .9
        self.state_size =state_size
        self.action_size =action_size
        self.env = env
        
        
    def initialize_state_actions(self ):
        ''' 
        Algorithm parameters: step size ↵2(0,1], small ">0 
        Initialize Q(s,a), for all s2S+,a2A(s), arbitrarily except that Q(terminal,·)=
        '''
        self.Q_vals= np.zeros((self.state_size, self.action_size))
                

    def choose_randomly(state, actions, period, args={}):
        return np.random.choice(actions)
    
    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def set_policy(self, policy, policy_args={}):
        self.policy = policy
        self.policy_args = policy_args
        
 
    
    
 
 
    def get_action(self, state, actions):
            '''
            Returns behavioural policy action
            which would be ε-greedy π policy, takes state and
            returns an action using this ε-greedy π policy
            '''
            if np.random.binomial(1, self.ε) == 1:
                #print(1)
                return np.random.choice(actions)
            else:
                A = np.ones(self.action_size, dtype=float) * self.ε / self.action_size
                best_action = np.argmax(self.Q_vals[state])
                #print(best_action)
                A[best_action] += (1.0 - self.ε)
                action = np.random.choice(np.arange(self.action_size), p=A)  
                return action
    
    def control(self):
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
        possible_actions = [0,1,2,3,4,5]
        #done=False
        print('***************************')
        state =self.env.reset()
        print(state)
        print('***************************')
        done=False
        while not done:
            action = self.get_action(state, possible_actions)
            # Take a step
            next_state, reward, done, details = self.env.step(action)
            next_state1=next_state
            # print(reward)
            
            # TD Update
             
            best_next_action = np.argmax(self.Q_vals[next_state]) 
            td_target = reward + (self.γ * self.Q_vals[next_state][best_next_action])
            td_delta =td_target - self.Q_vals[state][action]
            self.Q_vals[state][action]  +=  self.α * td_delta
            
            state=next_state
            
    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)