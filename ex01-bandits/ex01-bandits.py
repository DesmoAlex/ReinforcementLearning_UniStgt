import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    # iterate through all possible arms
    for a in possible_arms:
        # playing each arm once
        rewards[a] = bandit.play_arm(a)
        # is set to 1, because every arm is played once
        n_plays[a] = 1
        # is set to rewards[a] because n_play is 1 for every arm
        Q[a] = rewards[a]

    # Main loop
    while bandit.total_played < timesteps:
        # This example shows how to play a random arm:
        #a = random.choice(possible_arms)
        #reward_for_a = bandit.play_arm(a)
        # TODO: instead do greedy action selection
        # search indices with maximum estimate of action-value
        a = np.argmax(Q)
        # TODO: update the variables (rewards, n_plays, Q) for the selected arm
        # play greedy arm a
        rewards[a] += bandit.play_arm(a)
        # incremet greedy arm a
        n_plays[a] += 1
        # calculate new estimated action-value
        Q[a] = rewards[a]/n_plays[a]



def epsilon_greedy(bandit, timesteps):
    # define epsilon
    e = 0.1
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)
    # Initialize (same as in greedy)
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    # iterate through all possible arms
    for a in possible_arms:
        # playing each arm once
        rewards[a] = bandit.play_arm(a)
        # is set to 1, because every arm is played once
        n_plays[a] = 1
        # is set to rewards[a] because n_play is 1 for every arm
        Q[a] = rewards[a]


    while bandit.total_played < timesteps:
        # weigthed selection (greedy or not greedy)
        # if 0 go for random, (if 1 go for greedy)
        if random.random() > e:
            # search indices with maximum estimate of action-value
            a = np.argmax(Q)
            
        else:
            # TODO: instead do greedy action selection
            a = random.choice(possible_arms)
            # play greedy arm a

            
        # TODO: update the variables (rewards, n_plays, Q) for the selected arm
        rewards[a] += bandit.play_arm(a)
        # incremet greedy arm a
        n_plays[a] += 1
        # calculate new estimated action-value
        Q[a] = rewards[a]/n_plays[a]

        #reward_for_a = bandit.play_arm(0)  # Just play arm 0 as placeholder


def main():
    n_episodes = 10000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
