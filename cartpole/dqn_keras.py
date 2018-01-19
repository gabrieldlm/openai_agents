# No exp replay
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

agent = Sequential()
agent.add(Dense(16, activation = 'relu', input_dim = 4))
agent.add(Dense(16, activation = 'relu'))
agent.add(Dense(2,  activation = 'linear'))
adam = Adam(lr = 1e-3)
agent.compile(optimizer = adam, loss = 'mse')

env = gym.make('CartPole-v0')
eps = 10000
gamma = .99
exp_prob = 1

for ep in range(eps):
	total_reward = 0
	state = env.reset()
	state = np.reshape(state, [1, 4])
	done = False

	while not done:
		q = agent.predict(state)
		if np.random.rand() < exp_prob:
			action = np.random.randint(2)
		else:
			action = np.argmax(q)

		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, [1, 4])
		total_reward += reward

		Q = agent.predict(next_state)
		maxQ = np.max(Q)
		y = np.zeros((1,2))
		
		y[:] = q[:]

		if done:
			update = reward
		else:
			update = reward + gamma * maxQ

		y[0][action] = update

		agent.fit(state, y, epochs = 1, verbose = 0)

		if exp_prob > .1:
			exp_prob *= .995

		state = next_state

	print("Ep:{}, reward:{}".format(ep, int(total_reward)))

