import gym
import numpy as np

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
#env = gym.make("FrozenLake-v0", desc=custom_map3x3)
# TODO: Uncomment the following line to try the default map (4x4):
env = gym.make("FrozenLake-v0")

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8


""" This is a helper function that returns the transition probability matrix P for a policy """
def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """
def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    v = np.zeros(n_states)

    # Define Unit Matrix
    I = np.zeros((n_states, n_states))
    for i in range(n_states):
        I[i][i] = 1

    # subtract gamma*P from I
    Z = np.subtract(I, gamma*P)
    # inverse
    Z = np.linalg.inv(Z)
    # multiply inversed matrix with r-vector
    v = np.matmul(Z,r)


    # (P, r and gamma already given)
    return v


def bruteforce_policies():
    terms = terminals()
    optimalpolicies = []

    policy = np.zeros(n_states, dtype=np.int)  # in the discrete case a policy is just an array with action = policy[state]
    optimalvalue = np.zeros(n_states)
    
    # TODO: implement code that tries all possible policies, calculate the values using def value_policy. Find the optimal values and the optimal policies to answer the exercise questions.
    # loop trough all possible policies (TODO: except terminals)
    #lv0 state_0
    for a in range(4):
        policy[0] = a
        #lv1 state_1
        for b in range(4):
            policy[1] = b
            #lv2 state_2
            for c in range(4):
                policy[2] = c
                #lv3 state_3
                for d in range(4):
                    policy[3] = d
                    #lv4 state_4
                    for e in range(4):
                        policy[4] = e
                        #lv5 state_5
                        for f in range(4):
                            policy[5] = f
                            #lv6 state_6
                            for g in range(4):
                                policy[6] = g
                                #lv7 state_7
                                for h in range(4):
                                    policy[7] = h
                                    #lv8 state_8
                                    for i in range(4):
                                        policy[8] = i
                                        #lv 9 state_9
                                        for j in range(4):
                                            policy[9] = j
                                            #lv10 state_1ß
                                            for k in range(4):
                                                policy[10] = k
                                                #lv11 state_11
                                                for l in range(4):
                                                    policy[11] = l
                                                    #lv12 state_12
                                                    for m in range(4):
                                                        policy[12] = m
                                                        #lv13 state_4
                                                        for n in range(4):
                                                            policy[13] = n
                                                            print(policy)
                                                            policy_value = value_policy(policy)
                                                            if np.all(policy_value == optimalvalue):
                                                                optimalpolicies.append(policy)
                                                            if np.all(policy_value >= optimalvalue):
                                                                optimalvalue = policy_value
                    #                                           optimalpolicies = []
                                                                optimalpolicies.append(policy)







    print ("Optimal value function:")
    print(optimalvalue)
    print ("number optimal policies:")
    print (len(optimalpolicies))
    print ("optimal policies:")
    print (np.array(optimalpolicies))
    return optimalpolicies



def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=np.int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print (value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print (value_policy(policy_right))

    optimalpolicies = bruteforce_policies()


    # This code can be used to "rollout" a policy in the environment:
    """
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
