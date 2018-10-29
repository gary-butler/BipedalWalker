import gym
import numpy as np
import random

env = gym.make("BipedalWalker-v2")


#for state normalization
factors=[1/2*np.pi,5,2,2,1,
             0.5,1,0.3,1,1,
             0.5,1,0.25,1,2,
             1,1,1,1,1,
             1,1,1,1]

#this technique is slow to learn, so we need a lot of episodes
episodes = 100000

learning_rate = 1
epsilon = 0.8

#tracking our improvement
old_reward = -300
old_reward_total = -300

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
state_matrix = np.zeros(num_states)

#i've built a sort of neural net from scratch using numpy arrays
#this section initializes the parts of this neural net
action_max = np.zeros(num_actions)
hidden = np.zeros([3, 300, num_states, num_actions])
new_hidden = np.zeros([3, 300, num_states, num_actions])
best_hidden = np.zeros([3, 300, num_states, num_actions])
for layer in range(3):
    for steps in range(300):
        for item in range(num_states):
            for i in range(num_actions):
                hidden[layer, steps, item, i] = random.random() * 2 - 1
                new_hidden[layer, steps, item, i] = random.random() * 0.5 - 0.25
best_hidden = hidden

#main code block begins here
for i in range(episodes):

    #initialize variables for each episode
    reward_total = 0
    steps = 0
    done = False
    hidden = best_hidden
    state_matrix = env.reset() * factors #factors normalizes our state in one line
    
    while(done != True):

        #add some noise to our new hidden layers
        for item in range(num_states):
            for sub_item in range(num_actions):
                new_hidden[0, steps, item, sub_item] = hidden[0, steps, item, sub_item] + ((random.random() * 2 - learning_rate) * ((epsilon + ((episodes - i) / episodes)) / (301 - steps)))
                new_hidden[0, steps, item, sub_item] = np.clip(new_hidden[0, steps, item, sub_item], -1, 1)
                new_hidden[1, steps, item, sub_item] = hidden[1, steps, item, sub_item] + ((random.random() * 2 - learning_rate) * ((epsilon + ((episodes - i) / episodes)) / (301 - steps)))
                new_hidden[1, steps, item, sub_item] = np.clip(new_hidden[1, steps, item, sub_item], -1, 1)
                new_hidden[2, steps, item, sub_item] = hidden[2, steps, item, sub_item] + ((random.random() * 0.5 - 0.25) * ((epsilon + ((episodes - i)/episodes)) / (301 - steps)))
                new_hidden[2, steps, item, sub_item] = np.clip(new_hidden[2, steps, item, sub_item], -0.5, 0.5)
                
        #pass our state through the layers of the neural net
        layer2 = new_hidden[0, steps].T * state_matrix
        layer3 = layer2 * new_hidden[1, steps].T
        action = np.clip(np.dot(layer3, new_hidden[2, steps]), -1, 1)
        
        #choose the actions with the biggest impact, in testing this proved an effective choice
        for num in range(4):
            high = np.amax(action[num, :])
            low = np.amin(action[num, :])
            if (abs(high) > abs(low)):
                action[num] = high
            else:
                action[num] = low
        action = np.array((action[:, 0]))

        #take action
        state_matrix, reward , done, _ = env.step(action)

        state_matrix = state_matrix * factors #normalize our state in one shot
        reward_total += reward
        old_reward = reward
        steps += 1

        #exit if we are taking too long (maybe this could be increased, but it takes long time to compute)
        if(steps >= 300):
            done = True
    #update our neural net if it's the best
    if(reward_total > old_reward_total):
        best_hidden = new_hidden
        hidden = new_hidden
        old_reward_total = reward_total
    
    #keep track of our progress, no rendering for me, couldn't get it working
    if((i % 100) == 0):
        print(i, ": reward total : ", reward_total, " : best : ", old_reward_total, " : last action : ", action)

