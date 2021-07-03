
import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    # for simplicity devide calculations in two calcs for each action
    policy_l = sigmoid(  np.matmul(state, (theta[:,0] - theta[:,1]))  ) # calc see word doc
    policy_r = sigmoid(  np.matmul(state, (theta[:,1] - theta[:,0]))  )
    # return [0.5, 0.5]  # both actions with 0.5 probability => random
    return [policy_l, policy_r]


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)  #MEMO understand for exam
        
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def REINFORCE(env, alpha=0.005, gamma=0.9):
    theta = np.random.rand(4, 2)  # policy parameters
    ep_length = []
    mean_ep_length_100 = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        print("episode: " + str(e) + " length: " + str(len(states)))

        # TODO: keep track of previous 100 episode lengths and compute mean
        ep_length.append(len(states))
        mean_ep_length_100.append(np.mean(ep_length[-100:]))

        # TODO: implement the reinforce algorithm to improve the policy weights
        T = len(states) - 1
        for t in range(T):  # t to T-1, note range ends at T-1
                G = np.sum(
                        [gamma ** (k - t - 1) * rewards[k] # sum expression
                        for k in range(t + 1, T + 1)]  # sum indices, +1 on upper end bc range goes to upperend-1!
                )

                theta = theta + gamma**t * alpha * G * score_func(states[t], actions[t], theta)

    plt.plot(mean_ep_length_100)
    plt.show()

def score_func(s_t, a_t, theta):
    pi_l, pi_r = policy(s_t, theta)
    grad = np.empty((4, 2))

    if a_t == 0:
        grad_l = pi_r * s_t  #TODO sicherheitshalber policy ausschreiben
        grad_r = pi_r * -s_t
    else:
        grad_l = pi_l * s_t
        grad_r = pi_l * -s_t

    grad[:, 0] = grad_l
    grad[:, 1] = grad_r
    
    return grad



def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()