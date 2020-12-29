import numpy as np
import theano
import theano.tensor as T
import math
import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
import h5py
from keras.optimizers import Adamax, Nadam
import sys
from writeNNet import saveNNet
from matplotlib import plt

######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 10  # Epoch frequency of saving
totalEpochs = 200  # Total number of training epochs
trainingDataFiles = "../TrainingData/VertCAS_TrainingData_v2_%02d.h5"  # File format for training data
nnetFiles = "../networks/VertCAS_pra%02d_v%d_45HU_%03d.nnet"  # File format for .nnet files
##########################




# CONSTANTS
Rp = 4000
Rv = 575.4
pra = SCL1500
ra = CL1500
pd = 0
eps = 3

params = advisoryParams(ra, pd=pd, )

# Get variables from params dict:
w = params['w']
vlo = params['vlo']
alo = params['alo']
ahi = params['ahi']
wr = params['wr']
vlor = params['vlor']
ws = params['ws']
vlos = params['vlos']
eps = params['eps']
pd = params['pd']
# *****************************
v = [-30, -1]
vmin, vmax = v
hInitLower = 0
# traj = getNominalTraj(vmin,vmin,w*alo,0,pd,hInitLower,isLower=True
Hp = 100  # Height of NMAC
G = 32.2  # Gravitational acceleration

# CONSTANTS
HP = 100  # Height of NMAC
G = 32.2  # Graviational acceleration

'''
Represents quadratic boundaries that take the form:
    h = coeffs[0] + coeffs[1]*(t-minTau) + 0.5coeffs[2]*(t-minTau)^2
'''

# The previous RA should be given as a command line input
# implement safe regions


advisory = advisoryParams(ra, pd=pd, eps=eps)
_, _, boundMin, boundMax = getSafeable(advisory, v, worstCase=False)


def safe_region(r):
    """ Argument:
       r varies between 0nmi and 7nmi
       Hp between −4,000ft and 4,000ft
       Rv = between 0kts and 1,000kts
       V and vI between −5,000 and +5,000ft/min
       w is either -1 or 1
       The accelation ao is g/2 where g is the gravitational acceleration
    """

    """return:
    d, the vertical distance between the safe region and the intruder.
       If the intruder is in the safe region, d = 0.
       bound-1,
       bound-2,
       bound-3
       """

    minH = boundMin[0].getH_minTau()
    maxH = boundMax[0].getH_minTau()
    maxTau = boundMax[-1].getMaxTau()

    if -Rp <= r & r < -Rp - r * v * np.minimum(0, w * v) / alo:

        bound_1 = alo / 2.0 * np.square(r + Rp) + w * Rv * v * np.sum(r, Rp) - Rv * 2.0 * Hp

        if satisfiesBounds(bound_1 + boundMin, maxH, maxTau):
            d = maxH - bound_1

        return d

    elif -Rp - Rv * np.minimum(0, np.dot(w, v)) / alo <= r <= Rp - Rv * np.minimum(0, np.dot(w, v)) / alo:

        bound_2 = np.dot(w, maxH) < -np.square(np.minimum(0, np.dot(w, maxH))) / 2.0 * alo - Hp
        if satisfiesBounds(bound_2 + boundMin, maxH, maxTau):

            d = maxH - bound_2

        return d

    elif Rp - Rv * np.minimum(0, np.dot(w, v)) / alo < r <= Rp + Rv * np.maximum(0, w * np.subtract(vlo - v)) / alo:

        bound_3 = alo / 2.0 * np.square(r - Rp) + w * Rv * v * np.subtract(r, Rp) - Rv * 2.0 * Hp
        if satisfiesBounds(bound_3 + boundMin, maxH, maxTau):

            d = maxH - bound_3

        return d

# The previous RA should be given as a command line input

if len(sys.argv) > 1:
    pra = int(sys.argv[1])
    print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
    f = h5py.File(trainingDataFiles % pra, 'r')
    X_train = np.array(f['X'])
    Q = np.array(f['y'])
    means = np.array(f['means'])
    ranges = np.array(f['ranges'])
    min_inputs = np.array(f['min_inputs'])
    max_inputs = np.array(f['max_inputs'])

    N, numOut = Q.shape
    print("Setting up Model")

    # Asymmetric loss function
    lossFactor = 40.0
    lambd = 2
    
    def asymMSE(y_true, y_pred):
        distance = safe_region(20)
        # train the neural network by penalizing every distance that's grater than 0
        d = (y_true - y_pred) - tf.cast(lambd * tf.keras.backend.minimum(0, distance), dtype=tf.float32)
        maxes = tf.keras.backend.argmax(y_true, axis=-1)
        maxes_onehot = tf.keras.backend.one_hot(maxes, numOut)
        others_onehot = maxes_onehot - 1
        d_opt = d * maxes_onehot
        d_sub = d * others_onehot
        a = lossFactor * (numOut - 1) * (d_opt  2 + keras.backend.abs(d_opt))
        b = d_opt  tf.keras.backend.constant(2)
        c = lossFactor * (d_sub  2 + keras.backend.abs(d_sub))
        d = d_sub  tf.keras.backend.constant(2)
        loss = tf.keras.backend.switch(d_sub > 0, c, d) + tf.keras.backend.switch(d_opt > 0, a, b)
        return loss
if len(sys.argv) > 1:
    pra = int(float(sys.argv[1]))
    print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
    f = h5py.File(trainingDataFiles % pra, 'r')
    X_train = np.array(f['X'])
    Q = np.array(f['y'])
    means = np.array(f['means'])
    ranges = np.array(f['ranges'])
    min_inputs = np.array(f['min_inputs'])
    max_inputs = np.array(f['max_inputs'])

    N, numOut = Q.shape
    print("Setting up Model")

    # Asymmetric loss function
    lossFactor = 40.0

    # Define model architecture
    model = Sequential()
    model.add(Dense(hu, activation='relu', input_dim=4))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(numOut))
    opt = Nadam(lr=0.0003)
    model.compile(loss=asymMSE, optimizer=opt, metrics=['accuracy'])

    # Train and write nnet files
    epoch = saveEvery
    while epoch <= totalEpochs:
        model.fit(X_train, Q, epochs=saveEvery, batch_size=2 ** 8, shuffle=True)
        saveFile = nnetFiles % (pra, ver, epoch)
        saveNNet(model, saveFile, means, ranges, min_inputs, max_inputs)
        epoch += saveEvery

    plt.scatter()
