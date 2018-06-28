# OpenGym Seaquest-v0
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym Seaquest-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
# 
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, scipy , sys
from SumTree import SumTree
from keras import backend as K
from keras.callbacks import History
from decimal import Decimal
from matplotlib import pyplot as plt
import os.path

import tensorflow as tf


import socket

HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00025

#-------------------- UTILITIES -----------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, StateSize, ActionSize):
        self.StateSize = StateSize
        self.ActionSize = ActionSize

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network
        # if(os.path.exists("./DDQN_with_new_types.h5")):
        #     self.model.load_weights("DDQN_with_new_types.h5")
        #     self.model_.load_weights("DDQN_with_new_types.h5")
        #     print("loaded DDQN_with_new_types.h5 successfully") 
        # else:
        #     print("no save file")

    def _createModel(self):
        model = Sequential()
        model.add(Dense(output_dim=300, activation='relu', input_dim=StateSize))
        model.add(Dense(output_dim=600, activation='relu'))
        model.add(Dense(output_dim=ActionSize, activation='linear'))

        model.add(Dense(units=ActionSize, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        return self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.StateSize), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 0.01
MIN_EPSILON = 0.01

EXPLORATION_STOP = 10000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, StateSize, ActionSize):
        self.StateSize = StateSize
        self.ActionSize = ActionSize

        self.brain = Brain(StateSize, ActionSize)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            print("not my action")
            return random.randint(0, self.ActionSize-1)
        else:
            print("my action")
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.StateSize)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), self.StateSize))
        y = numpy.zeros((len(batch), self.ActionSize))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        return self.brain.train(x, y)
        

#-------------------- main ---------------------
host = '' 
port = 50000 
backlog = 5 
size = 1024 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.bind((host,port)) 
s.listen(backlog)

StateSize = 82
ActionSize = 9 

number_of_epochs = 0
time = 0
distance = 0
sign = 1
Episode_count = 0
Episodes = []
Rewards = []
Rewards_plot = []
Losses = []
arr_of_epochs = []
# hist = History()

tmp_state = [None] *StateSize
tmp_state_ = [None] *StateSize
State = [] 
State_ = []
a = 0
r = 0                           # immediate reward
R = 0                           # total reward
done = False                    # A flag to signal if the episode is done 

agent = Agent(StateSize, ActionSize)

client,address = s.accept()
print("client connected")

try:
    while True :

        client.send("Send the starting State\n".encode('utf-8'))
        response = str(client.recv(100000),"utf-8").rstrip('\r\n')
        if (response != "done" and response != "finished") :
            temp = response.split(",")
            for x in range(0,StateSize):
                tmp_state[x] = round(float(temp[x]),1)
            
            State = numpy.array(tmp_state)
        
        while True :
            a = agent.act(State)        #choosing the action

            client.send("do action ".encode('utf-8')+str(a).encode('utf-8')+"\n".encode('utf-8'))
            
            if (str(client.recv(1024),"utf-8").rstrip('\r\n') == "action done") :
                #print("action done")
                client.send("Send the starting State\n".encode('utf-8'))
            else :
                print("an error has occured while performing the action")
                sys.exit()

            response = str(client.recv(100000),"utf-8").rstrip('\r\n')
            if (response != "done" and response != "finished") :
                temp = response.split(",")
                for x in range(0,StateSize):
                    tmp_state_[x] = round(float(temp[x]),1)
            else :
                done = True
                
            State_ = numpy.array(tmp_state_)

            if (State_[79] == 1):
                sign = 1
            else:
                sign = -1

            if (done) :
                r = -5
            else :
                r = sign * (1 - 0.2 * sign * abs(math.sin(math.radians(State_[73]*10))) - 0.5 * State_[78]) * State_[72]

            i = 0
            while i < 72 :
                if(State_[i] == 0 and State_[i+1] > 0 and State_[72] > 5) :
                    r -= 0.5
                    break
                i += 2 

            if done :
                State_ = None
            print(response)
            if response == "finished" :
                r = 20
            
            print("Reward = " + str(r))

            agent.observe((State,a,r,State_))

            hist = agent.replay()
            # print(hist)

            Losses.append(hist.history['loss'])

            number_of_epochs += 1
            arr_of_epochs.append(number_of_epochs)
            print(number_of_epochs)

            # time += 0.2

            State = State_

            R += r

            r = 0

            if done :
                break
        
        print("total reward : ", R)

        # agent.brain.model.save("DDQN_with_new_types.h5")

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
    plt.figure(2)
    plt.plot(arr_of_epochs,Losses,'r')
    plt.title("The loss over the number of epochs")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

client.send("Bye!\n".encode('utf-8'))
client.close()
