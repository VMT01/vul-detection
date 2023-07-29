import os
from gensim.models.keyedvectors import Word2VecKeyedVectors
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from gensim.models import Word2Vec

import tensorflow as tf

################### Define constants ##################

VUL_DIR = '/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/Unprotected Suicide.csv'
NO_VUL_DIR = '/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/Contracts_No_Vul.csv'

VUL_RATIO = 1.0
NO_VUL_RATIO = 1.0

SPLIT_RATIO = 0.8 # Train vs Val data

MAX_FEATURES_PER_LINE = 1100
MAX_FEATURES_LINE = 5
MAX_FEATURES = MAX_FEATURES_PER_LINE * MAX_FEATURES_LINE

TITLE = 'Word-To-Vec_LSTM_Unprotected-Suicide'
EPOCHS = 10
BATCH_SIZE = 128

################### Reading & Dividing data ##################

def read_data(vul_ratio=0.1, no_vul_ratio=0.1):
    def format_bytecode(bytecode):
        bytecode = bytecode[:3*MAX_FEATURES].strip()
        if len(bytecode) == 3*MAX_FEATURES-1:
            return bytecode
        return bytecode + ' 00' * (MAX_FEATURES - (len(bytecode)+1)//3)

    df_vul = pd.read_csv(VUL_DIR, usecols=['BYTECODE']).dropna().drop_duplicates(subset=['BYTECODE']).sample(frac=vul_ratio)
    df_vul['BYTECODE'] = df_vul['BYTECODE'].apply(format_bytecode)
    df_vul['LABEL'] = 1.

    df_no_vul = pd.read_csv(NO_VUL_DIR, usecols=['OPCODE']).dropna().drop_duplicates(subset=['OPCODE']).sample(frac=no_vul_ratio)
    df_no_vul.rename(columns={'OPCODE':'BYTECODE'}, inplace=True)
    df_no_vul['BYTECODE'] = df_no_vul['BYTECODE'].apply(format_bytecode)
    df_no_vul['LABEL'] = 0.

    df = pd.concat([df_no_vul, df_vul]).sample(frac=1)

    #X = np.array([np.array(x.split()) for x in df['BYTECODE'].values])
    X = df['BYTECODE'].values
    y = df['LABEL'].values.astype(np.float32)

    return X, y

def divide_data(X, y):
    def split_data(data, ratio):
        index = int(len(data) * ratio)
        return data[:index], data[index:]

    X_train, X_val = split_data(X, SPLIT_RATIO)
    y_train, y_val = split_data(y, SPLIT_RATIO)
    
    X_train, X_test = split_data(X_train, SPLIT_RATIO)
    y_train, y_test = split_data(y_train, SPLIT_RATIO)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

################### Re-balance data modules ##################



################### Features extraction modules ##################

# Shape: [-1, MAX_FEATURES, 1]
def oneHotEncoding_module(X): 
    valid_tokens = { x:idx+1 for idx,x in enumerate(set(X.flatten())) }
    X = [[valid_tokens[_] for _ in x] for x in X]
    
    return np.expand_dims(X, 1)

# Shape: [-1, 1, features_len]
# Warning - Not test
def bagOfWord_module(X):
    v = CountVectorizer();
    X = v.fit_transform(X).toarray()

    # return np.reshape(X, (-1, MAX_FEATURES, len(v.get_feature_names_out())))
    return np.reshape(X, (-1, 1, X.shape[1]))

# Shape: [-1, 1, features_len]
def tfidf_module(X):
    v = TfidfVectorizer()
    X = v.fit_transform(X).toarray()

    # return np.reshape(X, (-1, MAX_FEATURES, len(v.get_feature_names_out())))
    return np.reshape(X, (-1, 1, X.shape[1]))

# Shape: [-1, ?]
# Warning - Not test
def word2vec_module(X):
    w2v_model = Word2Vec(X, window=5, min_count=5, workers=4)
    def vectorize(sentence):
        words_vecs = np.array([w2v_model.wv[word] for word in sentence if word in w2v_model.wv])
        if len(words_vecs) == 0:
            return np.zeros(100) # Why 100?
        return words_vecs.mean(axis=0)
    X = np.array([vectorize(x) for x in X])
        
    return np.reshape(X, (-1, 1, X.shape[1]))

################### Training models ##################

def BERT_model():
    return

def LSTM_model(input_shape):
    model = Sequential()

    model.add(LSTM(512, input_shape=input_shape, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True, activation='relu'))
    model.add(Dropout(0.1))
    model.add(LSTM(64, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model

################### Main ##################

def main(title, epochs=10, batch_size=64):
    try:
        os.rmdir(title)
        os.mkdir(title)    
    except FileNotFoundError:
        os.mkdir(title)

    X, y = read_data(vul_ratio=VUL_RATIO, no_vul_ratio=NO_VUL_RATIO)
    #X = oneHotEncoding_module(X)
    #X = bagOfWord_module(X)
    #X = tfidf_module(X)
    X = word2vec_module(X)
    
    # print(X.shape)
    # X = np.expand_dims(X, 1) # (-1, MAX_FEATURES, 1)

    input_shape = (X.shape[1], X.shape[2])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = divide_data(X, y)
    
    # clf = MultinomialNB()
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_val)
    # print('Training size = %d, accuracy = %.2f%%' % (X_train.shape[0], accuracy_score(y_val, y_pred)*100))

    model = LSTM_model(input_shape)
    history_logger = tf.keras.callbacks.CSVLogger(title + '/history.csv', separator=',')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[history_logger], validation_data=(X_val, y_val))
    model.save_weights(title + '/model.h5')
    
    y_pred = model.predict(X_test)
    y_pred = np.round(np.ravel(y_pred))
    
    print(classification_report(y_test, y_pred))
    
    # np.save(title + '/history.npy', history.history)

main(title=TITLE, epochs=EPOCHS, batch_size=BATCH_SIZE)

