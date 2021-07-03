import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    number_of_iterations = 0

    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r   
    # Emulate do-while:
    while True:
        delta = 0
        for s in range(len(V_states)):  #TODO vllt besser states aus env holen und darueber iterieren
            v = V_states[s]
            # search for max wrt action a:
            tmp_values = []
            for a in range(n_actions):
                tmp_values.append(perform_sum(s,a, V_states, gamma))
            V_states[s] = max(tmp_values)
            delta = max(delta, np.absolute(v - V_states[s]))
        number_of_iterations += 1
        # print(V_states)
        if (delta < theta):
            break
    print("Number of iterations to converge:")
    print(number_of_iterations)
    print("Optimal Value Function:")
    print(V_states)

    # Computing optimal policy:
    policy = np.zeros(n_states)
    for s in range(len(V_states)):
        tmp_values = []
        for a in range(n_actions):
            tmp_values.append(perform_sum(s,a, V_states, gamma))
        policy[s] = tmp_values.index( max(tmp_values) )

    return policy

def perform_sum(s,a, V_states, gamma): 
    # Bc of multiple nested loops, seperate function for value iteration step
    # performs:  sum_over_s'_r p*(r + gamma*V(s'))
    # s' = s_next
    sum_sr = 0
    for (p, s_next, r, is_terminated) in env.P[s][a]:
        sum_sr += p * ( r + gamma*V_states[s_next] )
    # print(sum_sr)
    return sum_sr


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
