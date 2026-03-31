import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
global action_values 
test = int(input("Enter num of bandits"))
action_values = []
for i in range(test):
    action_values.append(rng.normal())
def slot_value(n):
    return action_values[n] + rng.normal()
k_values = [0]*test
q_values = [5]*test
reward = 0
temp_reward = 0
i_values = []
avg_rew = []
for i in range(1000):
    max = 0
    for j in range(test):
        if(q_values[j]>q_values[max]):
            max = j
    temp_reward = slot_value(max)
    k_values[max] = k_values[max] + 1
    q_values[max] = q_values[max] + (1/k_values[max])*(temp_reward-q_values[max])
    reward = reward + temp_reward
    avg_rew.append(reward/(i+1))
    i_values.append(i)
plt.plot(i_values,avg_rew)
plt.show()
print(reward/1000)
print(q_values[0])
print(q_values[1])
