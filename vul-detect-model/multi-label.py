import polars as pl
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from keras import Model, Input, optimizers
from keras.layers import Dense, Dropout, LSTM, Embedding

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

# ====================================================

TOKEN = 5500
MAX_SEQ_LEN = 5500 * 3 - 1 # 5500 token
LABEL = [
    'Unprotected Suicide',
    'Unchecked call return value',
    'Ugradeable contract',
    'Timestamp dependence',
    'Outdated Solidity version',
    'Leaking Ether to arbitrary address',
    'Frozen Ether',
    'Delegatecall Injection',
    'Authentication through tx.origin',
]
VUL_PATH = '/home/bkcs/Desktop/multi classify/Data_Cleansing-002.csv'
EPOCHS = 1
BATCH_SIZE = 128

# ==========================================================

def format_bytecode(bytecode):
        bytecode = bytecode[:3*TOKEN].strip()
        if len(bytecode) == 3*TOKEN-1:
            return bytecode
        return bytecode + ' ##' * (TOKEN - (len(bytecode)+1)//3)

def data_loader():
    print('Reading data...')
    df = pl.scan_csv(VUL_PATH).collect().drop_nulls()

    X = df['BYTECODE'].apply(format_bytecode)
    y = df.drop(columns=['index', 'ADDRESS', 'BYTECODE', 'LABEL', 'LABEL_FORMAT'])

    del df # free memory

    return X.to_numpy(), y

def tfidf_module(X):
    print('TF-IDF')
    v = TfidfVectorizer()
    X = v.fit_transform(X).toarray()

    return X  

def build_model(shape, size):
    print('Building model...')
    input = Input(shape=(shape,))
    embedding = Embedding(shape * size, 100)(input)
    dropout1 = Dropout(0.2)(embedding)

    lstm1 = LSTM(256, return_sequences=True, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(lstm1)

    lstm2 = LSTM(128, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(lstm2)

    output1 = Dense(1, activation='sigmoid')(dropout3)
    output2 = Dense(1, activation='sigmoid')(dropout3)
    output3 = Dense(1, activation='sigmoid')(dropout3)
    output4 = Dense(1, activation='sigmoid')(dropout3)
    output5 = Dense(1, activation='sigmoid')(dropout3)
    output6 = Dense(1, activation='sigmoid')(dropout3)
    output7 = Dense(1, activation='sigmoid')(dropout3)
    output8 = Dense(1, activation='sigmoid')(dropout3)
    output9 = Dense(1, activation='sigmoid')(dropout3)

    model = Model(inputs=input, outputs=[output1, output2, output3, output4, output5, output6, output7, output8, output9])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['acc'])

    return model

if __name__ == "__main__":
    X, y = data_loader()

    X_tfidf = tfidf_module(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, train_size=0.8, shuffle=True)

    y1_train = y_train['Unprotected Suicide'].to_numpy()
    y1_test = y_test['Unprotected Suicide'].to_numpy()

    y2_train = y_train['Unchecked call return value'].to_numpy()
    y2_test = y_test['Unchecked call return value'].to_numpy()

    y3_train = y_train['Ugradeable contract'].to_numpy()
    y3_test = y_test['Ugradeable contract'].to_numpy()

    y4_train = y_train['Timestamp dependence'].to_numpy()
    y4_test = y_test['Timestamp dependence'].to_numpy()

    y5_train = y_train['Outdated Solidity version'].to_numpy()
    y5_test = y_test['Outdated Solidity version'].to_numpy()

    y6_train = y_train['Leaking Ether to arbitrary address'].to_numpy()
    y6_test = y_test['Leaking Ether to arbitrary address'].to_numpy()

    y7_train = y_train['Frozen Ether'].to_numpy()
    y7_test = y_test['Frozen Ether'].to_numpy()

    y8_train = y_train['Delegatecall Injection'].to_numpy()
    y8_test = y_test['Delegatecall Injection'].to_numpy()

    y9_train = y_train['Authentication through tx.origin'].to_numpy()
    y9_test = y_test['Authentication through tx.origin'].to_numpy()

    model = build_model(len(X_train[0]), len(X_train))
    model.load_weights('./model-lstm-new.h5')

    # model.save('./model-lstm.h5')
    y_pred = model.predict(X_test)
    y_pred = np.around(y_pred)

    y_test = y_test.to_numpy()
    # y_test = [y1_test,y2_test, y3_test,y4_test,y5_test,y6_test,y7_test,y8_test,y9_test]
    # y_test = np.argmax(y_test, axis=1)

    result = classification_report(y_test, y_pred, output_dict=True)
    print(result)
    # pl.DataFrame(result).write_csv('./result-new-lstm.csv')

