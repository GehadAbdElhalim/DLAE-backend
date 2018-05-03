import random, numpy, math, gym, sys
from keras import backend as K

import tensorflow as tf


import socket
#----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

#----------
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
        self.model_ = self._createModel()
        self.model.load_weights("DLAE-recording-weights.h5") 

    def _createModel(self):
        model = Sequential()
        model.add(Dense(output_dim=300, activation='relu', input_dim=StateSize))
        model.add(Dense(output_dim=600, activation='relu'))
        model.add(Dense(output_dim=ActionSize, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

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
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

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
            return random.randint(0, self.ActionSize-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # if self.steps % UPDATE_TARGET_FREQUENCY == 0:
        #     self.brain.updateTargetModel()

        # # debug the Q function in poin S
        # if self.steps % 100 == 0:
        #     S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
        #     pred = agent.brain.predictOne(S)
        #     print(pred[0])
        #     sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.StateSize)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.StateSize))
        y = numpy.zeros((batchLen, self.ActionSize))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

#-------------------- main ---------------------
host = '' 
port = 50000 
backlog = 5 
size = 1024 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.bind((host,port)) 
s.listen(backlog)

StateSize = 43 
ActionSize = 9 

tmp_state = [None] *43
tmp_state_ = [None] *43
State = [] 
State_ = []
a = 0
r = 0                           # immediate reward
R = 0                           # total reward
done = False                    # A flag to signal if the episode is done 

agent = Agent(StateSize, ActionSize)

while True :
    client,address = s.accept()
    print("client connected")

    #client.send("Hello!\n".encode('utf-8'))

    client.send("Send the starting State\n".encode('utf-8'))
    response = str(client.recv(100000),"utf-8").rstrip('\r\n')
    if (response != "done") :
        temp = response.split(",")
        for x in range(0,43):
            tmp_state[x] = float(temp[x])
        
        State = numpy.array(tmp_state)
        print(State)
        # print(response)
        # client.send("send next element\n".encode('utf-8'))
        # for x in range(0,43) :
        #     print("how")                                                    #receiving State
        #     data = str(client.recv(1024),"utf-8").rstrip('\r\n')
        #     print("1" + data)
        #     State.append(float(data))
        #     client.send("send next element\n".encode('utf-8'))
        #     #client.send("received".encode('utf-8')+data.encode('utf-8')+"\n".encode('utf-8'))

        print("done receiving the first state")
    
    while True :
        a = agent.act(State)        #choosing the action

        client.send("do action ".encode('utf-8')+str(a).encode('utf-8')+"\n".encode('utf-8'))
        # if (str(client.recv(1024),"utf-8").rstrip('\r\n') == "ok") :
        #     client.send(str(a).encode('utf-8'))
        #     print(a)
        if (str(client.recv(1024),"utf-8").rstrip('\r\n') == "action done") :
            print("action done")
            client.send("Send the starting State\n".encode('utf-8'))
        else :
            print("an error has occured while performing the action")
            sys.exit()

        response = str(client.recv(100000),"utf-8").rstrip('\r\n')
        if (response != "done") :
            temp = response.split(",")
            for x in range(0,43):
                tmp_state_[x] = float(temp[x])
            
        State_ = numpy.array(tmp_state_)
        print(State_)
        # if (str(client.recv(1024),"utf-8").rstrip('\r\n') == "done") :
        #     done = True
        # elif (str(client.recv(1024),"utf-8").rstrip('\r\n') == "ok") :   
        #     for y in State_ :                                               #receiving State_
        #         client.send("send next element \n".encode('utf-8'))
        #         data = str(client.recv(1024),"utf-8").rstrip('\r\n')
        #         y = float(data)
        #         client.send("received".encode('utf-8')+data.encode('utf-8')+"\n".encode('utf-8'))
            
        print("done receiving the state")

        r = State_[38] * math.cos(math.radians(State_[39])) - State_[38] * math.sin(math.radians(State_[39])) - 5 * State_[42]            #MUST BE REPLACED WITH THE REWARD FUNCTION

        if done :
            State_ = None
        
        agent.observe((State,a,r,State_))

        agent.replay()

        State = State_

        #State_ = []

        R += r

        if done :
            break
    
    print("total reward : ", R)

    agent.brain.model.save("DLAE-RL-weights.h5")

    client.send("Bye!\n".encode('utf-8'))
    client.close()
    break
