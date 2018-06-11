import numpy as np
import time
import cv2
import random
import copy
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D, AveragePooling3D, MaxPooling3D, GlobalMaxPooling3D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#flow [x, y, channel (angle, null, magnitude)]

#PROGRAM VARS
inputDim = (480, 640, 2)
kNumFrames = 5
kNumSamples = 10
kMotionThreshold = 75
kNumLabels = 26
runFusion = True

#staged files
files = []

#DATA
X_test = []
Y_test = []
X_train = []
Y_train = []

test_labels = []

alpha = ["a", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x","y","z"]

#optical flow data
sampledFlows = []
selectedFlows = []

def stageFiles():

    for character in alpha:
        for i in range(kNumFrames):

            #pick test samples
            if i == (ord(character) - 95)%4+2 or (character is "a" and i == 2):
                test_label = np.zeros((26,))
                test_label[ord(character)-97] = 1
                Y_test.append(test_label)
                X_test.append(character + str(i))
                continue

            files.append(character + str(i))

    Y_test = np.stack(test_labels)

def processVideoOpticalFlow(name):
    cap = cv2.VideoCapture(name)
    ret, frame1 = cap.read()

    flow = np.zeros_like(frame1)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    flow[...,2] = 255

    nSamples = 0
    while(nSamples != kNumSamples):

        ret, frame2 = cap.read()
        if(frame2 is None):
            cap.release()
            cap = cv2.VideoCapture(name)
            continue

        #print(name)
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        #encodings
        flow[...,0] = ang*180/np.pi/2 #magnitude
        flow[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) #angle

        if(bool(random.getrandbits(1))):
            sampledFlows.append(copy.deepcopy(flow))
            nSamples += 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = gray
    cap.release()

def applyThreshold():
    width = sampledFlows[0].shape[0]
    height = sampledFlows[0].shape[1]
    counts = []
    for i in range(kNumSamples):
        flow = sampledFlows[i]
        count = 0
        for x in range(25,50):
            for y in range(25, 50):
                if flow[x,y,0] > kMotionThreshold:
                    count+=1
        counts.append(count)
    counts.sort()
    for i in range(kNumFrames):
        selectedFlows.append(np.delete(sampledFlows[-i], 2, 2))

def generateData():
    allFlows = []
    labels = []
    for f in files:
        name = "/Users/diego/Desktop/cs230proj/alpha/" + f + ".mov"
        pos = ord(f[0]) - ord('a')
        y = np.zeros((26,))
        y[pos] = 1
        labels.append(y)

        processVideoOpticalFlow(name)

        sampledFlows = np.stack(sampledFlows)
        applyThreshold()
        selectedFlows = np.stack(selectedFlows)

        allFlows.append(selectedFlows)
        selectedFlows = []
        sampledFlows = []

    X_train = np.stack(allFlows)
    Y_train = np.array(labels)

    #GENERATE THE TEST SET
    for f in test:
        name = "ADD YOUR FILE PATH TO FILE" + f + ".mov"

        processVideoOpticalFlow(name)

        sampledFlows = np.stack(sampledFlows)
        applyThreshold()
        selectedFlows = np.stack(selectedFlows)

        X_test.append(selectedFlows)
        selectedFlows = []
        sampledFlows = []

    X_test = np.stack(X_test)
    Y_test = np.stack(Y_test)

def temporalModel(input_shape, classes = 26):
    bn_name_base = "flow"

    X_input = Input(input_shape)

    # Stage 1
    X = Conv3D(40, (3, 3, 3), strides = (2, 2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(X)

    X = Conv3D(2, (1, 1, 1), strides = (2, 2, 2), name = 'conv2', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((1, 2, 2), strides=(2, 2, 2))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='3Dfusion')

    return model

def runModel():
    model = temporalModel(input_shape = (5, 480, 640, 2), classes = 26)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs = 3, batch_size = 32)

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))



if __name__ == "__main__":

    print("Processing Data")
    stageFiles()
    generateData()
    print("Beginning Model")
    runModel()
