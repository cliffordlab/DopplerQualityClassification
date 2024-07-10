import pandas as pd 
import numpy as np
import tensorflow.keras as keras
import os
os.environ['TF_KERAS'] = '1'
import scipy
from scipy import signal
# from scipy.fftpack import fftshift
import scipy.io.wavfile
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed
import os.path
from tensorflow.keras.optimizers import Adam
# import keras
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.models import Model
import math
# Load dependencies
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense,Input,TimeDistributed,Conv2D,Flatten,BatchNormalization,Dropout,MaxPool2D
import random
from sklearn.model_selection import train_test_split
from Helper_functions import *
from tensorflow.keras.utils import to_categorical
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler

# Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)

#--------------------------------------------------------------------------------
#------------------------------------------------GPU----------------------------
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 } ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
#------------------------------------------------------------------------------

# function to train the model
def train(X_train,Y_train,fold):


    # Parameters
    freq_len=X_train.shape[2]
    time_len=X_train.shape[1]
    l2_reg = regularizers.l2(1e-2) 

    # --------------------------------- model -------------------------

    input_scalogram = Input(shape=(time_len,freq_len,1))
    y = Conv2D(32, (3,3),activation='relu',padding='same')(input_scalogram)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(2,2))(y)
    y = Dropout(0.25)(y)
    
    y = Conv2D(64, (3,3), activation='relu',padding='same')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(1,2))(y)
    y = Dropout(0.25)(y)
    
    y = Conv2D(128, (3,3), activation='relu',padding='same')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(1,2))(y)
    y = Dropout(0.25)(y)
    
    y = TimeDistributed(Flatten())(y)
    y = (GRU(50, return_sequences=True,activation='relu',name='GRU'))(y)
    y = Dropout(0.25)(y)
    
    y = Dense(50, activation='relu', name='dense')(y) 
    y = Dropout(0.25)(y) 
    
    attn,coeffs = HierarchicalAttentionNetwork(50,return_coefficients=True,name='attention')(y)

    preds = Dense(5, activation='softmax', kernel_regularizer=l2_reg)(attn)
    
    model = Model(input_scalogram, preds)
    optimizer = Adam(lr=0.001, decay=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    #------------------------------------train/validation split ----------------------------
    y_labels = np.argmax(Y_train, axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1, stratify=y_labels)

  
    #------------------------------------------training-------------------------------------
    dir='checkpoints/'
    create_directory_if_not_exists(dir)
    checkpoint_path = dir + "Weights_" + str(fold) + ".ckpt"
    es=EarlyStopping(monitor="val_loss", mode='min', verbose=1,patience=30)
    cp = ModelCheckpoint( filepath=checkpoint_path,verbose=1, save_weights_only=True, monitor='val_loss',save_best_only=True)   

    n_samples, time_len, freq_len, channels = X_train.shape
    X_train_2d = X_train.reshape((n_samples, time_len * freq_len * channels))   
    training_generator, steps_per_epoch = balanced_batch_generator(X_train_2d,Y_train, sampler=RandomOverSampler(), batch_size=128, random_state=1) #creating balanced batch
    reshaped_training_generator = ((batch_x.reshape((-1, time_len, freq_len, channels)), batch_y) for batch_x, batch_y in training_generator)
    
    history = model.fit(reshaped_training_generator,validation_data=(X_val,Y_val), steps_per_epoch = steps_per_epoch, epochs=500, verbose=1,callbacks=[cp,es])

    return model,checkpoint_path,history

# function to test the model
def test(x,model,checkpoint_path):
    model.load_weights(checkpoint_path)
    yhat=model.predict(x,batch_size=16)
    return yhat


#------------------------------------Loading the data ----------------------------
# Load data from an .npz file. 
#'tensor_all' contains the scalogram data for each 3.75s recording with shape (250,40).
# 'labels' contains the corresponding quality labels.
loaded_data = np.load('Data.npz')

# Expand dimensions of 'tensor_all' to include a channel dimension. Shape becomes (num_segments, 250, 40, 1).
tensor_rec=np.expand_dims(loaded_data['tensor_all'], axis=-1)

# Extract 'labels', which have the shape (num_segments, 1).
# Labels: 1: Good, 2: Poor, 3: Interference, 4: Talking, 5: Silent
label=loaded_data['label'] 

valid_indices = np.where(label[:,0] <= 5)
label_filtered = label[valid_indices]
label_filtered[:,0]=label_filtered[:,0]-1

# Convert labels to categorical format for 5 classes.
label_categorical=to_categorical(label_filtered[:,0], num_classes=5)

tensor_rec = tensor_rec[valid_indices]

# finding the mode of SQI in each recording
unique_visits = np.unique(label_filtered[:, 1])

modes = []
for visit in unique_visits:
    # Filter data for the current recording
    SQIs = label_filtered[label_filtered[:, 1] == visit][:, 0]
    
    # Find the mode
    mode = np.argmax(np.bincount(SQIs.astype(int)))
    modes.append([visit, mode])
modes=np.array(modes)    


# Stratify recordings based on the mode of quality labels in each recording.
fold=1
kf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1)
for train_visit_index, val_visit_index in kf.split(np.zeros(len(modes)), modes[:,1]):
    
    train_index = np.where(np.isin(label_filtered[:,1], modes[train_visit_index][:,0]))[0]
    val_index = np.where(np.isin(label_filtered[:,1], modes[val_visit_index][:,0]))[0]
    
    x_train=tensor_rec[train_index]
    y_train=label_categorical[train_index]
    
    x_val=tensor_rec[val_index]
    y_val=label_categorical[val_index]

    model,checkpoint_path,history=train(x_train,y_train,fold)
    plot_history(history, str(fold))

    y_train_hat=test(x_train,model,checkpoint_path)
    y_val_hat=test(x_val,model,checkpoint_path)

    dir='results/'
    create_directory_if_not_exists(dir)
    np.save(dir+'y_train'+str(fold),y_train)
    np.save(dir+'y_train_hat'+str(fold),y_train_hat)
    np.save(dir+'y_val'+str(fold),y_val)
    np.save(dir+'y_val_hat'+str(fold),y_val_hat)


    fold=fold+1    





