import numpy as np
import random
import operator
from lib import policies
from blackjack import env
import pandas as pd 
import pickle


class MonteCarloAgent():
    def __init__(self):
        self.name = 'MonteCarloAgent'
        self.terminal_state = np.array([0,0,0,0])
        self.Q_vals = {}
        self.C_vals = {}
        self.π = {}
        self.episode = []
        self.ε = 0.1
        self.γ = 1.0
        self.eps = dict({'S':[],'A':[],'probs':[],'R':[None]})
        self.target_policy = self.create_greedy_policy(self.Q_vals)
        self.behavior_policy=   policies.create_random_policy(2 ,None,None)
        
    def initialize_state_actions(self):
        all_states = [(p_sum, d_sum, ace)  for p_sum in range(12,23) for d_sum in range(1,23) for ace in  (True,False)]
        all_actions = [0,1]

        #all_sa = [(s, a) for a in all_actions for s in all_states]
 
        for s in all_states:
            s = tuple(s)
            for a in all_actions:
                self.Q_vals.setdefault(s, {})[a] = 0
                self.C_vals.setdefault(s, {})[a] = 0
                self.π[s]=2

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
    def control(self,  γ=1.0):
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
        possible_actions = [0,1]
        player_cards, dealer_cards = env.play_init()
        current_policy = self.behavior_policy
        r, l=env.play(player_cards, dealer_cards, current_policy)
        
        G = 0.0 # assumes all reward at end of episode
        W = 1.0
        T=len(l)
        for t in l[::-1]:
            T=T-1
            s, a  = t
            G = self.γ * G + r
            self.C_vals[s][a] += W
            self.Q_vals[s][a] += (W / self.C_vals[s][a]) * (G - self.Q_vals[s][a])
            self.π[s]  =np.argmax(self.target_policy(s))#np.argmax( list(self.Q_vals[s].values()))# np.argmax(self.Q_vals[s])
            r=0
            #print(self.π[s])
            #print("argmax: {},---{}".format(max(self.S_A_values[s].values()),self.S_A_values[s][a]))
            if a != self.π[s]:
                break
 
            W *= 1.0 /self.behavior_policy(s ,None,None)[a]
            #print(W)

    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)