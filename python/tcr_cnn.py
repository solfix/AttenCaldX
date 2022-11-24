###############################
# Author: Capas Peng          #
# Email: solfix123@163.com    #
###############################

import tensorflow as tf
from tensorflow import keras

def build_cnn_model(maxlen, n_feature):
    inputs = keras.layers.Input(shape=(maxlen,n_feature))
    
    conv1 = keras.layers.Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(inputs)
    conv3 = keras.layers.Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(inputs)
    conv5 = keras.layers.Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(inputs)
    conv7 = keras.layers.Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(inputs)
    conv9 = keras.layers.Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(inputs)
    
    pool1 = keras.layers.GlobalMaxPooling1D()(conv1)
    pool3 = keras.layers.GlobalMaxPooling1D()(conv3)
    pool5 = keras.layers.GlobalMaxPooling1D()(conv5)
    pool7 = keras.layers.GlobalMaxPooling1D()(conv7)
    pool9 = keras.layers.GlobalMaxPooling1D()(conv9)
    
    cat = keras.layers.concatenate([pool1, pool3, pool5, pool7, pool9], axis=1)
    
    dense = keras.layers.Dense(32, activation='sigmoid')(cat)
    out = keras.layers.Dense(1, activation='sigmoid')(dense)
    
    return keras.Model(inputs=inputs, outputs=out)
