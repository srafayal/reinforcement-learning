import pandas as pd 
import numpy as np


# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" 
ACTIONS = [ACTION_HIT, ACTION_STAND]


def dealer_policy(dealer_sum):
    return ACTION_HIT if dealer_sum<17 else ACTION_STAND
# get a new card
def get_card():
    card = np.random.randint(1, 14)
    return  card ,11 if card == 1 else min(card, 10)


def play_init():
    # sum of player
    data = {'card_name':[],'value':[]}
    player_cards = pd.DataFrame(data) 
    dealer_cards = pd.DataFrame(data)
    player_sum = 0

    
    while player_sum < 12:
        # if sum of player is less than 12, always hit
        card, new_card_value  = get_card()
        new_card = {'card_name':card, 'value':new_card_value}
        player_cards = player_cards.append(new_card, ignore_index=True)
        player_sum += new_card_value
    
    player_ace =    player_cards[player_cards.card_name== 1] 
    if sum(player_cards['value']) > 21 and len(player_ace)>0:
        for index, row in player_cards.iterrows():
            if row['card_name'] == 1:
                row['value']=1
            if sum(player_cards['value'])<= 21:
                break
    
    for i in range(0, 2):
        dealer_card, new_card_value  = get_card()
        new_card = {'card_name':dealer_card, 'value':new_card_value}
        dealer_cards = dealer_cards.append(new_card, ignore_index=True)
    
 
    
    dealer_ace =    dealer_cards[dealer_cards.card_name== 1] 
    if sum(dealer_cards['value']) > 21 and len(dealer_ace)>0:
        for index, row in dealer_cards.iterrows():
            if row['card_name'] == 1:
                row['value']=1
            if sum(dealer_cards['value'])<= 21:
                break
        
    
    
        
    return player_cards, dealer_cards


def play(player_cards, dealer_cards, policy_player,Q_vals=None,epsilon=.1):
    player_sum = sum(player_cards['value'])
    dealer_sum = sum(dealer_cards['value'])
    player_log=[]
    #print(dealer_cards['card_name'].iloc[0] )
    #print(dealer_cards)
    player_ace =    player_cards[player_cards.card_name== 1]
    
    
    #state = [usable_ace_player, player_sum, dealer_card1]

    
    
    
    # player's turn
    while True:
        # get action based on current sum
        state = (sum(player_cards['value']),  dealer_cards['card_name'].iloc[0], True if len(player_ace)>0 else False)
        probs = policy_player(state,Q_vals,epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)

#         # track player's card for importance sampling
        player_log.append([ ( sum(player_cards['value']),  dealer_cards['card_name'].iloc[0], True if len(player_ace)>0 else False ) , action])
         
        if action == ACTION_STAND:
            break
        # if hit, get new card
        card, new_card_value = get_card()
        new_card = {'card_name':card, 'value':new_card_value}
        player_cards = player_cards.append(new_card, ignore_index=True)
        
        # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
        # distinguish between having one ace or two.
        player_ace =    player_cards[player_cards.card_name== 1] 
        # If the player has a usable ace, use it as 1 to avoid busting and continue.
        if sum(player_cards['value']) > 21 and len(player_ace)>0:
            for index, row in player_cards.iterrows():
                if row['card_name'] == 1 and row['value']!=1:
                    row['value']=1
                if sum(player_cards['value'])<= 21:
                    break

#         # player busts
        player_sum = sum(player_cards['value'])
        if player_sum > 21:
#             print("P:{}".format(player_sum))
#             print('player busts')
#             print(player_cards)
            return  -1, player_log
        assert player_sum <= 21
    
#     print('dealer turn')
    # dealer's turn
    while True:
        # get action based on current sum
        action = dealer_policy(sum(dealer_cards['value']))
        
        if action == ACTION_STAND:
            break
            
        # if hit, get a new card     
        dealer_card, new_card_value  = get_card()
        new_card = {'card_name':dealer_card, 'value':new_card_value}
        dealer_cards = dealer_cards.append(new_card, ignore_index=True)
        
        ace =    dealer_cards[dealer_cards.card_name== 1] 
        
        # If the dealer has a usable ace, use it as 1 to avoid busting and continue.
        if sum(dealer_cards['value']) > 21 and len(ace)>0:
            for index, row in dealer_cards.iterrows():
                if row['card_name'] == 1 and row['value']!=1:
                    row['value']=1
                if sum(dealer_cards['value'])<= 21:
                    break
        # dealer busts
        dealer_sum = sum(dealer_cards['value'])
        if dealer_sum > 21:
#             print('dealer busts')
#             print("d:{}".format(dealer_sum))
#             print(dealer_cards)
            return  1, player_log

    
    # Reward Calculation
#     print("P:{}".format(player_sum))
#     print("d:{}".format(dealer_sum))
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return  1, player_log
    elif player_sum == dealer_sum:
        return  0, player_log
    else:
        return  -1, player_log