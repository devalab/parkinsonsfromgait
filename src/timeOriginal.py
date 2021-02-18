import os
import random
import shutil
import tqdm
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

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


temp = np.random.randn(100,12000,18)
label = 0
nums = np.random.randn(100,5)

# warmup
input_layer = Input(shape=(None,18))
x = SpatialDropout1D(0.2)(input_layer)
x = GaussianNoise(0.01)(x)
x = SeparableConv1D(32,7,activation="linear",kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation("elu")(x)
x = MaxPool1D(2,2)(x)
x = SeparableConv1D(32,5,activation="linear",kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation("elu")(x)
x = MaxPool1D(2,2)(x)
x = SeparableConv1D(64,3,activation="linear",kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation("elu",name="embedding")(x)
avg_pool = GlobalAvgPool1D()(x)
avg_pool = Dropout(0.25)(avg_pool)

prediction = Dense(1,activation="sigmoid",kernel_regularizer=regularizers.l2(0.001),name="det")(avg_pool)
model = Model(inputs=input_layer, outputs=prediction)


temp = np.random.randn(1000,12000,18)
pred = model.predict(temp)


# Measure
times = []
temp = np.random.randn(1000,6000,18)
t = time.clock()

pred = model.predict(temp)
#     prediction = np.mean(pred)

elapsed_time = time.clock() - t

print ("Execution time for a 2min recording at 50Hz: ",elapsed_time/1000, "seconds")

# Measure
times = []
temp = np.random.randn(1000,12000,18)
t = time.clock()

pred = model.predict(temp)
#     prediction = np.mean(pred)

elapsed_time = time.clock() - t

print ("Execution time for a 2min recording at 100Hz for Ablation: ",elapsed_time/1000, "seconds")