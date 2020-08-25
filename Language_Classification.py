# -*- coding: utf-8 -*-
import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import tensorflow
from sklearn.utils import class_weight
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

#function to initialize weights to each class to deal with class imbalance
def create_weights_matrix(labels): 
   
    
    weights_mapping = {0:1, 1:3.5, 2:1.4}
    
    if labels.ndim == 3:
        weights_matrix = np.zeros(labels.shape[0:2])
        for i,sample in enumerate(labels):
            for j,elem in enumerate(sample):
                weights_matrix[i,j] = weights_mapping[elem[0]]
    
    else:    
        weights_matrix = np.zeros(labels.shape[0])
        for i,sample in enumerate(labels):
            print(i)
            print(sample)
            weights_matrix[i] = weights_mapping[sample]
    
    return weights_matrix

#preprocess audiofiles and extract mfcc features for each audioclip
def preprocess(filepath,cat):
    for r,d,f in os.walk(filepath):
        list_lg=[]
        f=sorted(f)
        for filename in f:
            
            file=os.path.join(filepath,filename)
            #Sampling Rate 16000Hz
            y,sr=librosa.load(file,16000)
            #remove silence from audioclip
            interval=librosa.effects.split(y=y,top_db=45)
            y_nosilence=y[interval[0][0]:interval[0][1]]
            for i in interval[1:]:
                y_nosilence=np.concatenate([y_nosilence,y[i[0]:i[1]]])
            #extract mfcc features for each audioclip
            mat=librosa.feature.mfcc(y=y_nosilence,sr=sr,n_mfcc=64,n_fft=int((sr)*0.025),hop_length=int((sr)*0.01))
            list_lg.append(mat.T)
            print(y.shape,sr,mat.shape)
        
    array_lg=np.concatenate(list_lg,axis=0)
        
    train_seq_len=200   
    feature_dim=64
        
    n=array_lg.shape[0]%train_seq_len
    if n!=0:
        array_lg=array_lg[:-n, :]
    num_seq=int(array_lg.shape[0]/train_seq_len)
    
    array_lg=array_lg.reshape(num_seq,train_seq_len, feature_dim)
    y=np.full((num_seq,train_seq_len,1), cat)
    
    
    X_train, X_test, y_train, y_test = train_test_split(array_lg, y, test_size=0.2,shuffle=False,random_state=42)
    
    return X_train, X_test, y_train, y_test



#Design a model using a Gated Recurrent Unit for layers of the model
def train_model_design(X_train):
    training_in_shape = X_train.shape[1:]
    training_in = Input(shape=training_in_shape)
    foo1 = GRU(64, return_sequences=True, stateful=False)(training_in)
    foo = GRU(64, return_sequences=True, stateful=False)(foo1)
    training_pred = Dense(3,activation='softmax')(foo)
    
    training_model = Model(inputs=training_in, outputs=training_pred)
    training_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',sample_weight_mode="temporal",metrics=['accuracy'])
    training_model.summary()
    plot_model(training_model,to_file='model_image.png',show_shapes=True,show_layer_names=True)
    return training_model

#Design a streaming model using a Gated Recurrent Unit
def streaming_model_arch():
    streaming_in = Input(batch_shape=(1,None,64))  ## stateful ==> needs batch_shape specified
    foo1 = GRU(64, return_sequences=True, stateful=True )(streaming_in)
    foo = GRU(64, return_sequences=False, stateful=True)(foo1)
    streaming_pred = Dense(3,activation='softmax')(foo)
    streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)
    
    streaming_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    streaming_model.summary()
    plot_model(streaming_model,to_file='model_image.png',show_shapes=True,show_layer_names=True)
    return streaming_model
              




if __name__=='__main__':        
    
    #read training data files  and split into training and validation sets 
    X_train_md, X_test_md, y_train_md, y_test_md=preprocess('/home/ubuntu/train/train_mandarin',2)
    X_train_en, X_test_en, y_train_en, y_test_en =preprocess('/home/ubuntu/train/train_english',0)
    X_train_hi, X_test_hi, y_train_hi, y_test_hi =preprocess('/home/ubuntu/train/train_hindi',1)
    
    
    X_train=np.concatenate((X_train_md,X_train_hi,X_train_en),axis=0)
    y_train=np.concatenate((y_train_md,y_train_hi,y_train_en),axis=0)
    
    
    X_val=np.concatenate((X_test_md,X_test_hi,X_test_en),axis=0)
    y_val=np.concatenate((y_test_md,y_test_hi,y_test_en),axis=0)
    
    
    c = list(zip(X_train,y_train))
    np.random.shuffle(c)
    X_train, y_train = zip(*c)
    X_train=np.asarray(X_train)
    y_train=np.asarray(y_train)
    
    #shuffle traing and validation sets
    c = list(zip(X_val, y_val ))
    np.random.shuffle(c)
    X_val, y_val = zip(*c)
    X_val=np.asarray(X_val)                     
    y_val=np.asarray(y_val)
    
    #assign class weights to deal with class imbalance 
    weights_matrix=create_weights_matrix(y_train)
    
    #train both models for language classification task and save weights 
    training_model=train_model_design(X_train)
    training_model.fit(X_train, y_train,validation_data=(X_val,y_val), batch_size=16, epochs=10,shuffle=True,sample_weight=weights_matrix)#,class_weight=class_weight)
    training_model.save_weights('lang_rnn_weights.hd5', overwrite=True)
    
    streaming_model=streaming_model_arch()
    streaming_model.compile(loss='categorical_crossentropy', optimizer='adam')
    streaming_model.summary()
    streaming_model.load_weights('lang_rnn_weights.hd5')
    
    streaming_model.save('streaming_trained.hdf5')