import random, numpy, math, gym

import csv

from keras.models import Sequential
from keras.layers.merge import *
from keras.optimizers import *
from keras.models import Model
from keras.layers import Input, Dense

StateSize = 43
ActionSize = 9
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

model = Sequential()
model.add(Dense(output_dim=300, activation='relu', input_dim=StateSize))
model.add(Dense(output_dim=600, activation='relu'))
model.add(Dense(output_dim=ActionSize, activation='linear'))

opt = RMSprop(lr=LEARNING_RATE)
model.compile(loss=huber_loss, optimizer=opt)

#model.summary()


#reading the logfile of states and actions
states = []
actions = []
with open('state-action.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if (row[0] != "id0"):
            i = 0
            j = 2
            state = []
            action = []
            while (i < 36) :
                state.append(float(row[j]))
                i += 1
                j += 3
            for x in range(108 , 112) :
                state.append(float(row[x]))
            for y in range(112,114) :
                if row[y].strip() == "False" :
                    state.append(float(0))
                else :
                    state.append(float(1))

            state.append(float(row[114]))

            states.append(state)

            #action part
            action_number = 0
            if ((float(row[115]) > 0) and (float(row[116]) > 0)):
                action_number = 5
            elif ((float(row[115]) < 0) and (float(row[116]) > 0)):
                action_number = 6
            elif ((float(row[115]) > 0) and (float(row[116]) < 0)):
                action_number = 7
            elif ((float(row[115]) < 0) and (float(row[116]) < 0)):
                action_number = 8
            elif (float(row[115]) > 0):
                action_number = 1
            elif (float(row[115]) < 0):
                action_number = 2
            elif (float(row[116]) > 0):
                action_number = 3
            elif (float(row[116]) < 0):
                action_number = 4
            else :
                action_number = 0

            
            for z in range(0,9):
                if action_number == z :
                    action.append(float(1))
                else :
                    action.append(float(0))

            actions.append(action)

training_states = numpy.array(states)
training_actions = numpy.array(actions)

#training on the recorded sample 
model.fit(training_states, training_actions, batch_size=64, epochs=20, verbose=2)


#saving the weights
model.save("DLAE-recording-weights.h5")



