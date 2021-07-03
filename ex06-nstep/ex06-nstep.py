import gym
import numpy as np
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

def epsilon_greedy(env, epsilon, Q, S):
    if (random.random() < epsilon):
    # if False:
        A = random.choice(range(env.action_space.n))
    else:
        A = np.argmax(Q[S])
        # print(Q[S], A)
        # print()
    return A

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n,  env.action_space.n))  # (64, 4) 
    # Q = np.random.rand(env.observation_space.n,  env.action_space.n)  

    reached_goal = np.zeros(num_ep)
    for ep in range(num_ep):
        # S = []
        # A = []
        # S, A, R are 1dim in time:
        S = np.zeros(1000, dtype=int)
        A = np.zeros(1000, dtype=int)
        R = np.zeros(1000, dtype=int)

        # first state
        S[0] = env.reset()
        # S.append(env.reset())
        # S = np.append(S, env.reset())  # S0

        # Select first Action wrt Epsilon greedy
        A[0] = epsilon_greedy(env, epsilon, Q, S[0])


        t = 0
        T = 1000000
        while(True):  # iterating over t = 0, 1, 2 ...
            # print(t)
            if t < T:
                S[t+1], R[t+1], done, _ = env.step(A[t])
                # print(t, A[t])
                # env.render()

                if ep > num_ep-1:
                    env.render()
                
                # print("t = {}, S[t+1} = {} ", t, S[t+1])
                if R[t+1] == 1:
                    reached_goal[ep] = reached_goal[ep-1] + 1
                #     print("####")
                #     print("GOAL")
                #     print("####")
                else:
                    reached_goal[ep] = reached_goal[ep-1]

                if done:
                    # print("done!")
                    T = t + 1
                else:
                    # epsilon greedy:
                    A[t+1] = epsilon_greedy(env, epsilon, Q, S[t+1])
                    # print("t, A[t]: ", t, A[t])
            
            tau = t - n + 1

            if tau >= 0:
                # G = sum over rewards n steps in the future
                G = np.sum(
                    np.array(
                        [R[i] * pow(gamma, i-tau-1)  # sum expression
                        for i in range(tau + 1, min(tau+n, T) + 1)]  # sum indices, +1 on upper end bc range goes to upperend-1!
                        # min( tau+n=t+1,  t)
                    )
                )
                

                # add bootstrap to G
                if tau + n < T:
                    G += pow(gamma, n) * Q[ S[tau+n], A[tau+n] ]

                if R[t+1]!=0:
                    print([R[i] for i in range(tau + 1, min(tau+n, T)+1)])
                    print(G, R[t+1], alpha * (G - Q[ S[tau], A[tau] ]))
                
                # Q update
                Q[ S[tau], A[tau] ] += alpha * (G - Q[ S[tau], A[tau] ])

                #TODO need to do step "if pi is learned ..."?


            t += 1
            # print("" + str(t) + " " + str(tau) + " " + str(T))
            # end condition for while
            if tau == T - 1:
                # print(" tau == T -1 -> break")
                break

        if ep%10000 == 0:
            print_policy(Q, env)
            print()

    # fig = plt.figure()
    plt.plot(reached_goal)
    plt.show()
    # plt.imshow(reached_goal)
    # print(Q)
    


# env=gym.make('FrozenLake-v0', map_name="8x8")
env = gym.make('FrozenLake-v0')
# TODO: run multiple times, evaluate the performance for different n and alpha
env.render()
nstep_sarsa(env, n=5, epsilon=0.3, num_ep=50001)
