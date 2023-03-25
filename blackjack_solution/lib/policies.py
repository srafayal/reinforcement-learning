import numpy as np

# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return ACTION_HIT if player_sum<20 else ACTION_STAND

# # function form of behavior policy of player
# def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
#     return ACTION_STAND if  np.random.binomial(1, 0.5) == 1 else ACTION_HIT

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(  state,Q_vals, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        s =[]
        s.append(  state)  
        q_values = estimator.predict(np.array(s )) 
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        
        
        
        
        return A
    return policy_fn


def epsilon_greedy1(  action_size   ):
    """
    Returns behavioural policy action
    which would be ε-greedy π policy, takes state and
    returns an action using this ε-greedy π policy
    """
    def policy_fn(state,Q_vals,epsilon):
        A = np.ones(action_size, dtype=float) * epsilon / action_size
        best_action = np.argmax(Q_vals[ state ])
#         print(Q_vals[ state ])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
        
def epsilon_greedy(action_size):
    '''
    Returns behavioural policy action
    which would be ε-greedy π policy, takes state and
    returns an action using this ε-greedy π policy
    '''
    def policy_fn(state,Q_vals,epsilon):
#         np.random.seed(1)
        if np.random.uniform(0,1)<=epsilon:#np.random.binomial(1, epsilon) == 1:
            A =np.ones(action_size, dtype=float) * epsilon / action_size
            best_action = np.random.choice([0,1])
            A[best_action] += (1.0 - epsilon)
#             print(A)
            return  A
        else:
            A = np.ones(action_size, dtype=float) * epsilon / action_size
            best_action = np.argmax(Q_vals[ state ])
    #         print(Q_vals[ state ])
            A[best_action] += (1.0 - epsilon)
            return A  
    return policy_fn        
        
def create_random_policy( nA,Q_vals,epsilon):
        """
        Creates a random policy function.

        Args:
            nA: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """
        A = np.ones(nA, dtype=float) / nA
        def policy_fn(observation,Q_vals,epsilon):
            return A
        return policy_fn

 


 