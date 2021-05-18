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
        #TODO zero state?
        # print(obs)

        # done = False
        # reward = None
        # while not done:
        #     if obs[0] >= 20:  # poicy
        #         action = 0  # stick
        #     else:
        #         action = 1  # hit
        #     stateActionReward.append((obs, action, reward))
        #     obs, reward, done, _ = env.step(action)
        #     if done == True:

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

        # print(stateActionReward)
           
        stateActionReward.reverse()
        
        return_ = 0
        # no first visit check implemented: sum of cards is one part of the state. it's always increasing to we won't occur one specific state in an episode ever again
        for state, action, reward in stateActionReward:
            # print((reward))
            return_ += reward  # TODO useless because of no discounting return is always reward from termination state?
            if state in returns:
                returns[state].append(return_)
            else:
                returns[state] = []
                returns[state].append(return_)

            # calc values per state by averaging all returns
            values[state] = mean(returns[state])

    # for state in returns:
    #     values[state] = mean(returns[state])

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



    # ax.plot_surface(np.array(data_with_ace['x']), np.array(data_with_ace['y']), np.array(data_with_ace['z']))
    # tmp1, tmp2 = np.meshgrid(np.array(data_with_ace['x']), np.array(data_with_ace['y']))
    # ax.plot_surface(tmp1, tmp2, np.array(data_with_ace['z']))
    plt.show()


def MC_control_ES(env, episodes):
    returns = {}  # save return for each state
    Q_s_a = {}
    # create random policy: for all states put 0 or 1 randomly
    policy = {}
    for player_card_sum in range(4, 30):  #TODO check boundaries
        for dealer_card in range(1, 11):
            for usable_ace in [0, 1]:
                policy[(player_card_sum, dealer_card, usable_ace)] = random.randint(0,1)

    for i in range(episodes):
        stateActionReward = []
        obs = env.reset()

        # play game:
        done = False
        while not done:
            # if obs[0] >= 20:  # poicy
            #     action = 0  # stick
            # else:
            #     action = 1  # hit
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

            # policy improvement: select max value (over actions)
            try:
                if Q_s_a[(state, 0)] > Q_s_a[(state, 1)]:
                    policy[state] = 0
                else:
                    policy[state] = 1
            except KeyError:
                continue


            # if (state, 0) in Q_s_a:
            #     if (state, 1) in Q_s_a:
            #         policy[state] = max Q_s_a[(state, 1)]  # both actions available
            #     else:  # only action 0 available
            #         asdf
            # else:
            #     if (state, 1) in Q_s_a:  # only action 1


    ### extract data for plot
    # Without usable ace:
    data_with_ace = {}
    data_without_ace = {}
    data_with_ace['x'] = [] # x data
    data_with_ace['y'] = [] # y data
    data_with_ace['z'] = [] # z data
    data_without_ace['x'] = [] # x data2
    data_without_ace['y'] = [] # y data
    data_without_ace['z'] = [] # z data
    for (state, action), value in Q_s_a.items():
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
    plt.show()


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    
    # done = False
    # while not done:
    #     print("observation:", obs)
    #     if obs[0] >= 20:
    #         print("stick")
    #         obs, reward, done, _ = env.step(0)  # 0 is action stick
    #     else:
    #         print("hit")
    #         obs, reward, done, _ = env.step(1)  # 1 = hit
    #     print("reward:", reward)

    episodes = 100001
    # MC_prediction(env, episodes)
    MC_control_ES(env, episodes)
    

if __name__ == "__main__":
    main()
