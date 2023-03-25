import numpy as np
import random
import operator
import pandas as pd 
import pickle


class DynaQ():
    def __init__(self, state_size, action_size, env, ε=0.1, α = 0.5, planning_steps=25):
        self.name = 'DynaQ'
        self.terminal_state = np.array([0,0,0,0])
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
        
        self.state_size =state_size
        self.action_size =action_size
        self.env = env
        
        
    def initialize_state_actions(self):
        ''' 
        Initialize Q(s,a) and Model(s,a) for all s∈S and a∈A(s) 
        '''
         
        
        
 
        
        all_sa = [(s, a) for a in range(self.action_size) for s in range(self.state_size)]
        self.x_lookup = {(state, action): i for i, (state,action) in enumerate(all_sa)}
        self.y_lookup = {state:state for   state in range(self.state_size)}
        self.state_ix_lookup = {self.y_lookup[k]: k for k in self.y_lookup}
        
        
        self.Model = np.zeros((len(self.x_lookup), self.state_size))
        self.reward_table = {}
 
        self.Q_vals= np.zeros((self.state_size, self.action_size))

#     def choose_randomly(state, actions, period, args={}):
#         return np.random.choice(actions)
    
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
            
            #print(state)
            A = np.ones(self.action_size, dtype=float) * self.ε / self.action_size
            best_action = np.argmax(self.Q_vals[ state ])
            A[best_action] += (1.0 - self.ε)
            action = np.random.choice(np.arange(self.action_size), p=A)  
            return action
    
    def control(self,  γ=1.0):
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
        possible_actions = [0,1,2,3,4,5]
        #done=False
        #(a) S current (nonterminal) state 
        state =self.env.reset()
 
        done=False
        while not done:
            # (b) A <- ε-greedy(S,Q)
            action = self.get_action(state, possible_actions)
           # print(action)
      
            # (c) Take action A; observe resultant reward, R, and state, S'
            next_state, reward, done, details = self.env.step(action)
            #print("s:{}  a:{},  r:{}".format(state,action,reward))
            nexts=next_state
            #(d) Q(S,A)<-Q(S,A)+ α[R + γ* max_a Q(S',a)-Q(S,A)] 
            # Q-Learning update
            
            best_next_action = np.argmax(self.Q_vals[ next_state ]) 
            target = reward + (γ * self.Q_vals[next_state][best_next_action])
            qUpdate =target - self.Q_vals[state][action]
            
            #print("s:{},  b:{},  a:{},  nb:{} ,r:{}".format(state,self.Q_vals[state][action],action,self.α * qUpdate,target))
            self.Q_vals[state][action]  +=  self.α * qUpdate
            
            # feed the model with experience
            #(e) Model(S,A) R,S' (assuming deterministic environment) 
            self.Model[self.x_lookup[(state, action)], next_state]+= 1
            try:
                self.reward_table[(state, action, next_state)] += \
                self.α * (reward - self.reward_table[state, action, next_state])
            except KeyError:
                self.reward_table[(state, action, next_state)] = reward
            
            
            for sweeps in range(self.planning_steps):
                #S<-random previously observed state 
                state_ = random.choice(range(self.state_size))
                #A<-random action previously taken in S
                action_ = random.choice(range(self.action_size))
                
                #R,S' <- Model(S,A) 
                p_sas = self.Model[self.x_lookup[(state_, action_)], :]
                 
                if np.sum(p_sas) == 0:
                    continue
                else:
                    # Choose randomly from the next states based on their probability of being visited but ignoring terminal
                    p_sas = p_sas / np.sum(p_sas)
                    next_state_ = self.state_ix_lookup[np.random.choice(range(p_sas.size), p=p_sas)]
                    #print(state_)
                    reward_ = self.reward_table[(state_, action_, next_state_)]
                
                #Q(S,A)<-Q(S,A)+ α[R + γ* max_a Q(S',a)-Q(S,A)]
 
                best_next_action_ = np.argmax(self.Q_vals[ next_state_ ]) 
                target_ = reward_ + (γ * self.Q_vals[next_state_][best_next_action_])
                qUpdate_ =target_ - self.Q_vals[state_][action_]
                self.Q_vals[state_][action_]  +=  self.α * qUpdate_
                #print(state)
                
            state=nexts
            #print(state)
            
    def save(self, name):
        f = open(name,"wb")
        pickle.dump(self.Q_vals,f)
        f.close()
    def load(self, name):
        self.Q_vals = pd.read_pickle(name)