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
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    delta = theta
    steps = 0
    policy = np.zeros(n_states)

    # loop:
    # stop loop if delta < theta 
    while theta <= delta:
    	# initialize delta is 0
    	steps +=1
    	delta = 0
    	# iterate trough all possible states
    	for s in range(n_states):
    		# copy v_states into v 
    		v = list(V_states)
    		# calc for every action and safe highest value function in V_states
    		v_actions = np.zeros(n_actions)
    		for a in range(n_actions):
    			for i in env.P[s][a]:
    				v_actions[a] += i[0]*(i[2]+gamma*v[i[1]])
    				#DEBUG print("state: ", s," action a: ",a,"probability: ", i[0], " next state: ", i[1], " reward: ", i[2], " is terminal_ ", i[3])
    		V_states[s] = max(v_actions)
    		policy[s] = np.argmax(v_actions)
    		#DEBUG print(V_states[s])
    		#DEBUG print(v[s])
    		# calc difference
    		dif = abs(v[s]-V_states[s])
    		#DEBUG print("state:", s ,"dif: ", dif)
    		if dif > delta:
    			delta = dif
    		#DEBUG print("state:", s, "new delta:", delta)

    #return optimal value for debugging
    return policy
   

    # calc policy
    #return policy

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
