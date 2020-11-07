import numpy as np

k = 10
epsilon = 0.1

bandits = []
means = np.random.uniform(size=k, low=-5, high=5)
for i in range(k):
    bandits.append((means[i], 1))


def bandit(i):
    mu, sigma = bandits[i]
    return np.random.normal(mu, sigma)


q = np.zeros(k)
n = np.zeros(k)

# Training
for i in range(1000):
    # Sample random action
    if np.random.uniform() < epsilon:
        action = np.random.randint(0, k)
    # Sample from best actions
    else:
        action = np.random.choice(np.argwhere(q == np.amax(q))[0])
    reward = bandit(action)
    n[action] = n[action] + 1
    q[action] = q[action] + (1 / n[action]) * (reward - q[action])

print('Bandits distribution (mu, sigma):')
for i in range(k):
    print(f'#{i} ({bandits[i][0]:.2f}, {bandits[i][1]:.2f})')

print('\nValue function:')
print(q)
print('\nAction counts:')
print(n)
