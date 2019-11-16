#!/usr/bin/env python
from __future__ import print_function,division

import argparse
import sklearn
import skimage
from skimage import transform, color, exposure
# from skimage.transform import rotate
# from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras import backend as K
import tensorflow as tf


GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 6400. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_MEMORY = 10000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
RUNNING_OBSERVATION = 999999999
SAVING_FREQ = 300
NUM_ACTIONS = 2
RUNNING_MODEL_FILE = "model74200.h5"
TRAINING_MODEL_FILE = None

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames
def saveModel(model,t):
    print("Model being saved")
    model.save_weights(str("model"+str(t)+"h5"), overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

def imageProcessing(image):

    x_t1 = skimage.color.rgb2gray(image)
    x_t1 = skimage.transform.resize(x_t1, (80, 80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    x_t1/=255.0
    print("***************dividing")
    x_t1 = np.reshape(x_t1, (1, x_t1.shape[0], x_t1.shape[1], 1))
    print("***************reshaping")
    return x_t1 # 1x80x80x1

def buildModel(num_actions):
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8),  padding ='same',input_shape=(img_rows,img_cols,img_channels),activation="relu"))  #80*80*4
    model.add(Conv2D(64, kernel_size = (4, 4), padding='same',activation="relu"))
    model.add(Conv2D(64, kernel_size = (3, 3),  padding='same',activation="relu"))
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(Dense(num_actions))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("Model construction complete")

    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    buffer = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    print(x_t.shape)
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    print(type(x_t),"**************************")
    x_t = x_t / 255.0
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = np.reshape(s_t,(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])) #1*80*80*4

    print(args["mode"])
    if args['mode'] == 'Run':
        print("Running mode")
        OBSERVE = RUNNING_OBSERVATION    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Loading from trained model")
        model.load_weights(RUNNING_MODEL_FILE)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Trained model load successfully")
    else:                       #We go to training mode
        print("Training mode")
        if TRAINING_MODEL_FILE:
            model.load_weights(TRAINING_MODEL_FILE)
        else: 
            model = buildModel(NUM_ACTIONS)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Trained model load successfully")
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        random_action_list = [0,0,0,0,0,0,0,0,1] # make it 7 to 1
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                
                action_index = random.choice(random_action_list)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1
        if t>OBSERVE:
            EPSILON_TRAIN = 0.8
            epsilon = min(EPSILON_TRAIN,epsilon)
        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE and t<=EXPLORE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = imageProcessing(x_t1_colored)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in buffer
        buffer.append((s_t, action_index, r_t, s_t1, terminal))
        if len(buffer) > REPLAY_MEMORY:
            buffer.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(buffer, BATCH)

            #Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)
            loss += model.train_on_batch(state_t, targets)
        s_t = s_t1
        t+=1

        # save progress every 10000 iterations
        if t % SAVING_FREQ == 0 and t>OBSERVE:
            saveModel(model,t)
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index,\
              "/ REWARD", r_t, "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildModel(NUM_ACTIONS)
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    #
    # sess = tf.Session()
    #
    # K.set_session(sess)
    main()
