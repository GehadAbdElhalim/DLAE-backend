"""
solving pendulum using actor-critic model
"""

import gym , math , socket , numpy , sys
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, sess):
		self.sess = sess

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.tau   = .125

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=100000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, 2]) # where we will feed de/dC (from critic)

		# print(self.env.observation_space.shape)

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #		

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		state_input = Input(shape=[82])
		h1 = Dense(300, activation='relu')(state_input)
		h2 = Dense(600, activation='relu')(h1)
		output = Dense(2, activation='tanh')(h2)
		
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=[82])
		state_h1 = Dense(300, activation='relu')(state_input)
		state_h2 = Dense(600)(state_h1)
		
		action_input = Input(shape=[2])
		action_h1    = Dense(600)(action_input)
		
		merged    = Add()([state_h2, action_h1])
		merged_h1 = Dense(600, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
				reward += self.gamma * future_reward
			# action = numpy.reshape(action,2)

			reward = np.array(reward)
			reward = np.reshape(reward,(1,1))
			self.critic_model.fit([cur_state, action], reward, verbose=0)
		
	def train(self):
		batch_size = 32
		if len(self.memory) < batch_size:
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)
		self._train_critic(samples)
		self._train_actor(samples)

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_critic_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.critic_target_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.critic_target_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return numpy.array([[random.uniform(-1,1),random.uniform(-1,1)]])
		else:
			print(self.actor_model.predict(cur_state))
			print("hi")
			return self.actor_model.predict(cur_state)

def main():
	sess = tf.Session()
	K.set_session(sess)
	actor_critic = ActorCritic(sess)

	num_trials = 10000
	trial_len  = 500

	host = '' 
	port = 50000 
	backlog = 5 
	size = 1024 
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
	s.bind((host,port)) 
	s.listen(backlog)

	StateSize = 82
	ActionSize = 2 

	number_of_epochs = 0
	time = 0
	distance = 0
	sign = 1
	Episode_count = 0
	Episodes = []
	Rewards = []
	Rewards_plot = []

	tmp_state = [None] *StateSize
	tmp_state_ = [None] *StateSize
	State = [] 
	State_ = []
	a = 0
	r = 0                           # immediate reward
	R = 0                           # total reward
	done = False                    # A flag to signal if the episode is done 

	client,address = s.accept()
	print("client connected")
	try:
		while True :

			client.send("Send the starting State\n".encode('utf-8'))
			response = str(client.recv(100000),"utf-8").rstrip('\r\n')
			if (response != "done") :
				temp = response.split(",")
				for x in range(0,StateSize):
					tmp_state[x] = round(float(temp[x]),1)
				
				State = numpy.array(tmp_state)
				State = numpy.reshape(State, (1,82))
			
			while True :
				a = actor_critic.act(State)        #choosing the action
				a = numpy.array(a)
				a = numpy.reshape(a,(1,2))
				print(numpy.shape(a))

				client.send(str(a[0][0]).encode('utf-8')+",".encode('utf-8')+str(a[0][1]).encode('utf-8')+"\n".encode('utf-8'))
				
				if (str(client.recv(1024),"utf-8").rstrip('\r\n') == "action done") :
					print("action done")
					client.send("Send the starting State\n".encode('utf-8'))
				else :
					print("an error has occured while performing the action")
					sys.exit()

				response = str(client.recv(100000),"utf-8").rstrip('\r\n')
				if (response != "done") :
					temp = response.split(",")
					for x in range(0,StateSize):
						tmp_state_[x] = round(float(temp[x]),1)
				else :
					done = True
					
				State_ = numpy.array(tmp_state_)
				State_ = numpy.reshape(State_, (1,82))
				print(numpy.shape(State_))

				if (State_[0][79] == 1):
					sign = 1
				else:
					sign = -1

				if (done) :
					r = -5
				else :
					r = sign * (1 - 0.2 * sign * abs(math.sin(math.radians(State_[0][73]*10))) - 0.5 * State_[0][78]) * State_[0][72]

				i = 0
				while i < 72 :
					if(State_[0][i] == 6 and State_[0][72] > 5) :
						r -= 0.5
						break
					i += 2 

				print("Reward = " + str(r))
				
				actor_critic.remember(State, a, r, State_, done)
				actor_critic.train()

				number_of_epochs += 1
				print(number_of_epochs)

				# time += 0.2

				State = State_

				R += r

				r = 0

				if done :
					break
			
			print("total reward : ", R)

			actor_critic.actor_model.save("A3C.h5")

			client.send("restart\n".encode('utf-8'))

			if(str(client.recv(1024),"utf-8").rstrip('\r\n') == "oksh") :
				Episode_count += 1
				Episodes.append(Episode_count)
				Rewards.append(R)
				R = 0
				time = 0
				distance = 0
				done = False
			else :
				break
	finally :
		for j in range(1,len(Rewards)+1) :
			sum = 0
			for m in range(0,j) :
				sum += Rewards[m]
			Rewards_plot.append(sum/(j+1))

		plt.figure(1)
		plt.plot(Episodes,Rewards_plot,'b')
		plt.title("The average reward over the number of episodes")
		plt.xlabel("Episodes")
		plt.ylabel("Reward")

		client.send("Bye!\n".encode('utf-8'))
		client.close()


if __name__ == "__main__":
	main()