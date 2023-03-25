import numpy as np
import random
import operator
import pandas as pd 
import pickle
import sys

class Sarsa():
    def __init__(self, state_size, action_size,n_episodes, env):
        self.name = 'Sarsa'
        self.Q_vals = {}
        self.epsilon_start=1.0
        self.epsilon_end=0.1
        self.epsilon_decay_steps=50000

        self.alpha = 0.9 # Sarsa step size
        self.state_size =state_size
        self.action_size =action_size
        self.env = env
 
        self.n_episodes=n_episodes
        
    def initialize_state_actions(self ):
        ''' 
        Algorithm parameters: step size ↵2(0,1], small ">0 
        Initialize Q(s,a), for all s2S+,a2A(s), arbitrarily except that Q(terminal,·)=
        '''
        self.Q_vals= np.zeros((self.state_size, self.action_size))

 
    def get_action(self, state, epsilon):
            '''
            Returns behavioural policy action
            which would be ε-greedy π policy, takes state and
            returns an action using this ε-greedy π policy
            '''
             
            #print(state)
            if np.random.uniform(0, 1) <= epsilon: 
                return np.random.choice(np.arange(self.action_size))
            else:
                A = np.ones(self.action_size, dtype=float) * epsilon / self.action_size
                best_action = np.argmax(self.Q_vals[state])
                A[best_action] += (1.0 - epsilon)
                action = np.random.choice(np.arange(self.action_size), p=A)  
                return action
    
    def control(self,  gamma=0.9):
        '''
        Initialize S 
        Choose A from S using policy derived from Q (e.g., ε-greedy) 
        Loop for each step of episode: 
            Take action A, observe R, S' 
            Choose A' from S' using policy derived from Q (e.g., ε-greedy) 
            Q(S,A)<-Q(S,A)+ α[R + γQ(S',A')-Q(S,A)]
            S<-S'; A<-A';
        until S is terminal
        '''
        epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)
        epsilon=1
        total_t=0
        for e in range(self.n_episodes): # iterate over new episodes of the game
            if e % 1 == 0:
                print("\rEpisode {}/{}. ...{}".format(e, self.n_episodes,epsilon), end="")
                sys.stdout.flush()
 
            #done=False
            state =self.env.reset()
            epsilon = epsilons[min(total_t, self.epsilon_decay_steps-1)]  
            action = self.get_action(state, epsilon)

            done=False
            total_t+= 1 
            while not done:
                epsilon = epsilons[min(total_t, self.epsilon_decay_steps-1)]  
                # Take a step
                next_state, reward, done, details = self.env.step(action)
                # Pick the next action
                next_action = self.get_action(next_state, epsilon)

                # TD Update
                self.Q_vals[state][action]  =self.Q_vals[state][action]   +   self.alpha * (reward + (gamma * self.Q_vals[next_state][next_action]) -self.Q_vals[state][action])

                state=next_state
                action=next_action
                
                

    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)