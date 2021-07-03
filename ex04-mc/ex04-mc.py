import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mean

def MC_prediction(env, episodes):
    returns = {}  # save return for each state
    values = {}

    for i in range(episodes):
        stateActionReward = []
        obs = env.reset()

        # play game according to given policy:
        done = False
        while not done:
            if obs[0] >= 20:  # poicy
                action = 0  # stick
            else:
                action = 1  # hit
            obs, reward, done, _ = env.step(action)

            # save stateActionReward
            stateActionReward.append((obs, action, reward))  #TODO offset?!!!
           
        stateActionReward.reverse()
        
        return_ = 0
        # no first visit check implemented: sum of cards is one part of the state. it's always increasing to we won't occur one specific state in an episode ever again
        # solution: My thoughts were correct! no loop in environment!
        for state, action, reward in stateActionReward:
            return_ += reward  # TODO useless because of no discounting return is always reward from termination state?
            if state in returns:
                returns[state].append(return_)
            else:
                returns[state] = []
                returns[state].append(return_)

            # calc values per state by averaging all returns
            values[state] = mean(returns[state])

    ### extract data for plot
    # Without usable ace:
    data_with_ace = {}
    data_without_ace = {}
    data_with_ace['x'] = [] # x data
    data_with_ace['y'] = [] # y data
    data_with_ace['z'] = [] # z data
    data_without_ace['x'] = [] # x data
    data_without_ace['y'] = [] # y data
    data_without_ace['z'] = [] # z data
    for state, value in values.items():
        if state[0] > 11 and state[0] < 22:  # only print relevant area
            if state[2] == True:
                data_with_ace['x'].append(state[0])  # sum of player cards
                data_with_ace['y'].append(state[1])  # dealer card
                data_with_ace['z'].append(value)
            else:
                data_without_ace['x'].append(state[0])  # sum of player cards
                data_without_ace['y'].append(state[1])  # dealer card
                data_without_ace['z'].append(value)
    # plots
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter3D(data_without_ace['x'], data_without_ace['y'], data_without_ace['z'])
    # ax.scatter3D(data_with_ace['x'], data_with_ace['y'], data_with_ace['z'])
    plt.show()


###########################################################################################################33
def MC_control_ES(env, episodes):
    returns = {}  # save return for each state
    
    # create random policy: for all states put 0 or 1 randomly
    # create random state calue function Q_s_a
    policy = {}
    Q_s_a = {}
    for player_card_sum in range(4, 32):  #TODO check boundaries
        for dealer_card in range(1, 11):
            for usable_ace in [0, 1]:
                policy[(player_card_sum, dealer_card, usable_ace)] = random.randint(0, 1)
                for action in [0, 1]:
                    # Q_s_a[((player_card_sum, dealer_card, usable_ace), action)] = random.uniform(-1, 1)
                    Q_s_a[((player_card_sum, dealer_card, usable_ace), action)] = 0
    # NOTIZ works why better with list comperehension, sth like:
    # states_list = [(i, j, True) for j in range(10, 0, -1) for i in range(12, 22)]
    # states_list.extend([(i, j, True) for j in range(10, 0, -1) for i in range(12, 22)])

    for i in range(episodes):
        stateActionReward = []
        obs = env.reset()  # tutor says that this reset might be enough for exploring starts: every state has non-zero probability, but wihtout having the same transition probability for
        # each state it takes a unrealistiv long time!

        # Better something like that:
        

        ### play game:
        done = False
        while not done:
            action = policy[obs]
            obs, reward, done, _ = env.step(action)
            # save stateActionReward
            stateActionReward.append((obs, action, reward))  #TODO offset?!!!
           
        stateActionReward.reverse()
        
        return_ = 0
        # no first visit check implemented: sum of cards is one part of the state. it's always increasing to we won't occur one specific state in an episode ever again
        for state, action, reward in stateActionReward:
            # print((reward))
            return_ += reward  # TODO useless because of no discounting return is always reward from termination state?
            if (state, action) in returns:
                returns[(state, action)].append(return_)
            else:
                returns[(state, action)] = []
                returns[(state, action)].append(return_)

            # calc values per state by averaging all returns
            Q_s_a[(state, action)] = mean(returns[(state, action)])

            # policy improvement: select max value greedily (over all possible actions)
            if Q_s_a[(state, 0)] > Q_s_a[(state, 1)]:
                policy[state] = 0

        ### print policy
        if (i % 100000 == 0):
            print("iteration: ", i)
            print("player_card_sum -> action | ...")
            for state, action in policy.items():
                if state[0] > 12 and state[0] < 22 and state[2] == False:  # only print relecant states
                    print(state[0], "->", action, end="  |  ")
            print()


    # ### extract data for plot
    # # Without usable ace:
    # data_with_ace = {}
    # data_without_ace = {}
    # data_with_ace['x'] = [] # x data
    # data_with_ace['y'] = [] # y data
    # data_with_ace['z'] = [] # z data
    # data_without_ace['x'] = [] # x data
    # data_without_ace['y'] = [] # y data
    # data_without_ace['z'] = [] # z data
    # for (state, action), value in Q_s_a.items():
    #     if state[2] == True:
    #         data_with_ace['x'].append(state[0])  # sum of player cards
    #         data_with_ace['y'].append(state[1])  # dealer card
    #         data_with_ace['z'].append(value)
    #     else:
    #         data_without_ace['x'].append(state[0])  # sum of player cards
    #         data_without_ace['y'].append(state[1])  # dealer card
    #         data_without_ace['z'].append(value)
    # # plots
    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')
    # ax.scatter3D(data_with_ace['x'], data_with_ace['y'], data_with_ace['z'])
    # plt.show()



def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)

    episodes = 50000
    # MC_prediction(env, episodes)
    MC_control_ES(env, episodes)
    

if __name__ == "__main__":
    main()
