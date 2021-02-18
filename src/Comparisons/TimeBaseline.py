import os
import random
import shutil
import tqdm
import numpy as np
import pandas as pd
import copy
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras import layers, optimizers

import warnings
warnings.filterwarnings("ignore")


# inference time measurement on single core
import tensorflow as tf
from tensorflow.keras import backend as K

num_cores = 1


config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

session = tf.Session(config=config)
K.set_session(session)





# Helper Functions

def window(samples,feature,label,cut_length=100):
    inputs = []
    features = []
    labels = []
    
    for i in range(len(samples)):
        sample = samples[i]
        cut = int(cut_length/2)
        for j in range(int(len(sample)/cut)):
            if (j+2)*cut>=len(sample):
                break
            inputs.append(sample[j*cut:(j+2)*cut,:])
            features.append(feature[i])
            labels.append(label[i])
            
    inputs = np.stack(inputs)
    features = np.array(features)
    labels = np.array(labels)
    
    return inputs, features, labels

def pad(samples):
    lengths = [len(i) for i in samples]
    max_len = max(lengths)
    for i in range(len(samples)):
        pad_len = max_len - lengths[i]
        samples[i] = np.pad(samples[i],((0,pad_len),(0,0)),"wrap")
    return np.stack(samples)
    
def get_from_dict(dictionary, keys):
    output = []
    for i in keys:
        output += dictionary[i]
    return output

def get_best_model(path):
    models = os.listdir(path)
    accuracy = {}
    for i in models:
        info = i.split("-")
        try:
            accuracy[float(info[-1][:-5])][float(info[-2])] = i
        except:
            accuracy[float(info[-1][:-5])] = {}
            accuracy[float(info[-1][:-5])][float(info[-2])] = i
    best_acc = max(accuracy.keys())
    best_loss = min(accuracy[best_acc].keys())
    model_path = accuracy[best_acc][best_loss]
    
    model = tf.keras.models.load_model(path+"/"+model_path)
    return model

class VariancePooling(tf.keras.layers.Layer):
    def __init__(self, ):
        super(VariancePooling, self).__init__()

    def call(self, x):
        return tf.math.reduce_std(x,axis=1)
    
    
    
def conv1D_full():
    '''
    :return: 1 branch of the parallel Convnet
    '''
    input1 = Input(shape=(100, 1))
    x = Conv1D(filters=8, kernel_size=3, activation='selu', padding='valid')(input1)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = layers.MaxPooling1D(2)(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = Conv1D(filters=16, kernel_size=3, activation='selu', padding='valid')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='elu')(x)
    model = Model(input1, x)
    rms = optimizers.RMSprop(lr=0.001, decay=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
#     print(model.summary())
    return model


def multiple_cnn1D(nb=18):
    '''
    :param nb: number of features ( indicates the number of parallel branches)
    :return:
    '''
    inputs = Input(shape=(100,nb))
    outputs = []
    for i in range(nb):
        x = inputs[:,:,i]
        model = conv1D_full()
        x = tf.expand_dims(x,axis=-1)
        x = model(x)
        outputs.append(x)
    x = concatenate(outputs,axis=-1)
    x = Dropout(0.5)(x)
    x = layers.Dense(100, activation='selu')(x)
    x =  Dropout(0.5)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.5)(x)
    answer = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, answer)
    opt = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     print(model.summary())
    return model

def baseline(np=18):
        # Define Model
    input_layer = Input(shape=(None,18))
    x = SpatialDropout1D(0.2)(input_layer)
    x = Conv1D(32,7,activation="linear")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool1D(2,2)(x)
    x = Conv1D(32,5,activation="linear")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool1D(2,2)(x)
    x = Conv1D(64,3,activation="linear")(x)
    x = BatchNormalization()(x)
    x = Activation("elu",name="embedding")(x)
    avg_pool = GlobalAvgPool1D()(x)
    avg_pool = Dropout(0.25)(avg_pool)
    
    prediction = Dense(1,activation="sigmoid",name="det")(avg_pool)
    model = Model(inputs=input_layer, outputs=prediction)
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
          loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
          metrics=['accuracy'])
    return model

import time
testSeq = np.random.randn(100,12000,18)
testLabel = np.random.randn(100)
testNum = np.random.randn(100,5)

# warmup
full_model = baseline()
for i in range(len(testSeq)):
    pred = full_model.predict(window([testSeq[i]],[testNum[i]],[testLabel[i]])[0])
    prediction = np.mean(pred)
#     predictions.append(prediction)
#     gold.append(testLabel[i])



# Measure
times = []
t = time.clock()

for i in range(len(testSeq)):
    pred = full_model.predict(window([testSeq[i]],[testNum[i]],[testLabel[i]])[0])
    prediction = np.mean(pred)

elapsed_time = time.clock() - t
print ("Execution time for a 2min recording : ",elapsed_time/100, "seconds")