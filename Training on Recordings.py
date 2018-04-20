import random, numpy, math, gym

from keras.models import Sequential
from keras.layers.merge import *
from keras.optimizers import *
from keras.models import Model
from keras.layers import Input, Dense

StateSize = 41
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

model.summary()


#reading the logfile of states and actions




#training on the recorded sample where x is the states and y are the actions
#model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)


#saving the weights
#model.save("DLAE-recording-weights.h5")



