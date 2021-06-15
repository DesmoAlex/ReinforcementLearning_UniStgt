import gym
import numpy as np
import matplotlib.pyplot as plt
import random


def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def policy_q(env,q,s, epsilon):
    if random.random() > epsilon:
        # greedy
        a = np.argmax(q[s[0]][s[1]])
    
    else:
        # random
        a = np.random.randint(env.action_space.n)
    return a

def q_learning(env):
    # QLearning algorithm
    nep = 5000
    n_ep = 0

    #calc bins for aggregating
    bins_x = np.arange(-1.2,0.6,0.05)
    bins_v = np.arange(-0.07,0.07,0.005)
    nstate_x = bins_x.size
    #strange bug
    nstate_x = nstate_x
    nstate_v = bins_v.size

    epsilon = 0.01
    gamma=0.9
    alpha=0.5
    #init q arbitrairily
    q = np.random.uniform(0,1,(nstate_x,nstate_v,env.action_space.n))
    # set Terminal states to 0
    q[nstate_x-2:][:][:] = 0

    success = np.zeros(nep)
    steps = np.zeros(nep)




    while n_ep < nep:
        s_con = env.reset()
        s=np.zeros(2)
        # digitize space
        s[0] = np.digitize(s_con[0], bins_x)
        #digitze velocity
        s[1] = np.digitize(s_con[1], bins_v)
        s = s.astype(int)
        print(s)

        done = False
        while not done:
            #env.render()
            # Choose A from s using policy derived from q (epsilon greedy)
            a = policy_q(env,q,s,epsilon)
            s_con,r,done,info = env.step(a)
            steps[n_ep] += 1
            if done:
                success[n_ep:] += success[n_ep]+1
            # digitize space
            s_= np.zeros(2)
            s_[0] = np.digitize(s_con[0], bins_x)
            #digitze velocity
            s_[1] = np.digitize(s_con[1], bins_v)
            s_ = s_.astype(int)
            #update q 
            q[s[0]][s[1]][a] += alpha*(r + gamma*np.max(q[s_[0]][s_[1]]) - q[s[0]][s[1]][a])
            s=s_
        if (n_ep%20)==0:
            v = np.zeros((nstate_x,nstate_v))
            for a in range(nstate_x):
                for b in range(nstate_v):
                    v[a][b] = np.max(q[a][b])
            print(v)
                    

        n_ep += 1
        print("end of episode" + str(n_ep+1))
    

    plt.plot(success)
    plt.show()

    plt.plot(steps)
    plt.show()
        




def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    #random_episode(env)
    q_learning(env)
    env.close()


if __name__ == "__main__":
    main()
