import numpy as np
import random
import operator
import pandas as pd 
import pickle
import sys


class MonteCarloAgent():
    def __init__(self, state_size, action_size, n_episodes, env):
        self.name = 'MonteCarloAgent'
        self.terminal_state = np.array([0,0,0,0])
        self.Q_vals = {}
        self.C_vals = {}
        self.pie = {}
 
        self.gamma = 1.0

        self.target_policy = self.create_greedy_policy(self.Q_vals)
        self.behavior_policy=   self.create_random_policy(action_size)
        self.state_size =state_size
        self.action_size =action_size
        self.env = env
        self.n_episodes=n_episodes
    def initialize_state_actions(self):
             
        for s in range(self.state_size):
            for a in range(self.action_size):
                self.Q_vals.setdefault(s, {})[a] = 0
                self.C_vals.setdefault(s, {})[a] = 0
                self.pie[s]=0   
  
        
    def create_random_policy(self, nA):
        """
        Creates a random policy function.

        Args:
            nA: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """
        A = np.ones(nA, dtype=float) / nA
        def policy_fn(observation):
            return A
        return policy_fn
    
    def create_greedy_policy(self,Q):
        """
        Creates a greedy policy based on Q values.

        Args:
            Q: A dictionary that maps from state -> action values

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            A = np.zeros_like(list(Q[state]), dtype=float)
            best_action = np.argmax(list(Q[state].values()))

            A[best_action] = 1.0
            return A
        return policy_fn
    
    
    def control(self ):
        '''
        Performs MC control using episode list [ S0 , A0 , R1, . . . , ST −1 , AT −1, RT , ST ]
        G ← 0
        W ← 1
        For t = T − 1, T − 2, . . . down to 0:
            G ← γ*G + R_t+1
            C(St, At ) ← C(St,At ) + W
            Q(St, At ) ← Q(St,At) + (W/C(St,At))*[G − Q(St,At )]
            π(St) ← argmax_a Q(St,a) (with ties broken consistently)
            If At != π(St) then exit For loop
            W ← W * (1/b(At|St))        
        '''  
        for e in range(self.n_episodes): # iterate over new episodes of the game
            if e % 1 == 0:
                print("\rEpisode {}/{}".format(e, self.n_episodes), end="")
                sys.stdout.flush()
                
            done=False
            state =self.env.reset()
            episode = []
            
            while not done:
                probs = self.behavior_policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, details = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            G = 0.0 # assumes all reward at end of episode
            W = 1.0
            T=len(episode)
            for t in episode[::-1]:
                s, a, r = t
                G = self.gamma * G + r
                self.C_vals[s][a] += W
                self.Q_vals[s][a] += (W / self.C_vals[s][a]) * (G - self.Q_vals[s][a])


                self.pie[s] =np.argmax(self.target_policy(s))
                if a != self.pie[s]:
                    break
                W *= 1.0 /self.behavior_policy(s )[a]
            
    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)