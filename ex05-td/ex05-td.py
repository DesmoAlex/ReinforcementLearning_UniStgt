import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓',u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in ['H', 'G']:
            policy[idx] = u'·'
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) 
        for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper', 
               extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(V[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1,1]])
    down = np.array([[0, 0], [0.5, 0.5], [1,0]])
    left = np.array([[0, 0], [0.5, 0.5], [0,1]])
    right = np.array([[1, 0], [0.5, 0.5], [1,1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]
    
    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0,vmax=.6)
    
    ax.imshow(np.zeros(dims), origin='upper', extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3-x), 1, 1, color=cmap(.0)))
            plt.text(y+0.5, dims[0]-x-0.5, '{:.2f}'.format(.0),
                horizontalalignment='center', 
                verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3-x]), color=cmap(Q[s][a])))
            plt.text(y+pos[a][0], dims[0]-1-x+pos[a][1], '{:.2f}'.format(Q[s][a]), 
                     horizontalalignment='center', verticalalignment='center',
                    fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])


def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):  #TODO implement decreasing epsilon
    # zero initialisation:
    Q = np.zeros((env.observation_space.n,  env.action_space.n))  # (16, 4)

    # # random initialisation:
    # Q = np.random.rand(env.observation_space.n,  env.action_space.n)
    # # set terminal states to 0:
    # Q[5] *= 0
    # Q[7] *= 0
    # Q[11] *= 0
    # Q[12] *= 0
    # Q[-1] *= 0

    # # This is some starting point performing random walks in the environment:
    # for i in range(num_ep):
    #     s = env.reset()
    #     done = False
    #     while not done:
    #         a = np.random.randint(env.action_space.n)
    #         s_, r, done, _ = env.step(a)

    steps_per_episode = []
    average_steps_per_episode = []
    for i in range(num_ep):
        s = env.reset()

        ### 1. policy evaluation with epsilon-greedy:
        if (random.random() < epsilon):
        # if False:
            a = random.choice(range(env.action_space.n))
        else:
            a = np.argmax(Q[s])
        
        
        done = False
        # episode:
        steps_per_episode_counter = 0
        reached_goal = False
        while not done:
            # take action a, observe R, s_
            s_, r, done, _ = env.step(a)

            ### 1. policy evaluation with epsilon-greedy:
            # choose a_ from s_ using epsilon-greedy
            if (random.random() < epsilon):
                a_ = random.choice(range(env.action_space.n))
            else:
                a_ = np.argmax(Q[s_])

            ### 2. policy improvement:
            # calc Q_s_a incrementally
            Q[s][a] = Q[s][a] + alpha * ( r + gamma*Q[s_][a_] - Q[s][a] )
            s = s_
            a = a_

            steps_per_episode_counter += 1
            
            if r == 1:
                # env.render()
                reached_goal = True
                # print(r, reached_goal)
            
            
        
        # if reached_goal:
        #     steps_per_episode.append(steps_per_episode_counter)
        # else:
        #     steps_per_episode.append(0)
        steps_per_episode.append(steps_per_episode_counter)
        average_steps_per_episode.append(np.average(steps_per_episode))

        # print_policy(Q, env)
        # print(steps_per_episode_counter)
        # print()

    # print(steps_per_episode)
    # print(len(steps_per_episode))

    ### plotting average episode length
    # fig = plt.figure()
    # plt.plot(range(len(steps_per_episode)), steps_per_episode)
    plt.plot(average_steps_per_episode)
    # plt.draw()  #NOTE let's computation continue

    return Q


def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n,  env.action_space.n))
    steps_per_episode = []
    average_steps_per_episode = []
    for i in range(num_ep):
        s = env.reset()
        
        done = False
        # episode:
        steps_per_episode_counter = 0
        reached_goal = False
        while not done:
            ### 1. policy evaluation with epsilon-greedy:
            # choose a_ from s_ using epsilon-greedy
            if (random.random() < epsilon):
                a = random.choice(range(env.action_space.n))
            else:
                a = np.argmax(Q[s])

            ### 2. policy improvement:
            # take action a, observe R, s_
            s_, r, done, _ = env.step(a)

            # calc Q_s_a incrementally
            Q[s][a] = Q[s][a] + alpha * ( r + gamma*max(Q[s_]) - Q[s][a] )
            s = s_

            steps_per_episode_counter += 1
            
            if r == 1:
                # env.render()
                reached_goal = True
                # print(r, reached_goal)
            
            
        
        # if reached_goal:
        #     steps_per_episode.append(steps_per_episode_counter)
        # else:
        #     steps_per_episode.append(0)

        steps_per_episode.append(steps_per_episode_counter)
        average_steps_per_episode.append(np.average(steps_per_episode))

    plt.plot(average_steps_per_episode)

    return Q


env=gym.make('FrozenLake-v0')
# env=gym.make('FrozenLake-v0', is_slippery=False)
# env=gym.make('FrozenLake-v0', map_name="8x8")

# print("Running sarsa...")
# Q = sarsa(env, epsilon=0.1, num_ep=int(5e4))
# plot_V(Q, env)
# plot_Q(Q, env)
# print_policy(Q, env)
# plt.show()

print("Running qlearning")
Q = qlearning(env, num_ep=int(5e4))
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()
