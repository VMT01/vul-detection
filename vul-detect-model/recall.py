import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from gensim.models import Word2Vec

VUL_DIR = '/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/Unprotected Suicide.csv'
NO_VUL_DIR = '/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/Contracts_No_Vul.csv'
MAX_FEATURES_PER_LINE = 1100
MAX_FEATURES_LINE = 5
MAX_FEATURES = MAX_FEATURES_PER_LINE * MAX_FEATURES_LINE

vuls = [
	'Authentication through tx.origin',
	'Delegatecall Injection',
	'Frozen Ether',
	'Leaking Ether to arbitrary address',
	'Outdated Solidity version',
	'Timestamp dependence',
	'Ugradeable contract',
	'Unchecked call return value',
	'Unprotected Suicide'
]

titles = [
	'One-Hot-Encoding_LSTM_Authentication-through-tx-origin',
	'One-Hot-Encoding_LSTM_Delegatecall-Injection',
	'One-Hot-Encoding_LSTM_Frozen-Ether',
	'One-Hot-Encoding_LSTM_Leaking-Ether-to-arbitrary-address',
	'One-Hot-Encoding_LSTM_Outdated-Solidity-version',
	'One-Hot-Encoding_LSTM_Timestamp-dependence',
	'One-Hot-Encoding_LSTM_Ugradeable-contract',
	'One-Hot-Encoding_LSTM_Unchecked-call-return-value',
	'One-Hot-Encoding_LSTM_Unprotected-Suicide',
	###########################################################
	'Bag-Of-Word_LSTM_Authentication-through-tx-origin',
	'Bag-Of-Word_LSTM_Delegatecall-Injection',
	'Bag-Of-Word_LSTM_Frozen-Ether',
	'Bag-Of-Word_LSTM_Leaking-Ether-to-arbitrary-address',
	'Bag-Of-Word_LSTM_Outdated-Solidity-version',
	'Bag-Of-Word_LSTM_Timestamp-dependence',
	'Bag-Of-Word_LSTM_Ugradeable-contract',
	'Bag-Of-Word_LSTM_Unchecked-call-return-value',
	'Bag-Of-Word_LSTM_Unprotected-Suicide',
	###########################################################
	'TF-IDF_LSTM_Authentication-through-tx-origin',
	'TF-IDF_LSTM_Delegatecall-Injection',
	'TF-IDF_LSTM_Frozen-Ether',
	'TF-IDF_LSTM_Leaking-Ether-to-arbitrary-address',
	'TF-IDF_LSTM_Outdated-Solidity-version',
	'TF-IDF_LSTM_Timestamp-dependence',
	'TF-IDF_LSTM_Ugradeable-contract',
	'TF-IDF_LSTM_Unchecked-call-return-value',
	'TF-IDF_LSTM_Unprotected-Suicide',
	###########################################################
	'Word-To-Vec_LSTM_Authentication-through-tx-origin',
	'Word-To-Vec_LSTM_Delegatecall-Injection',
	'Word-To-Vec_LSTM_Frozen-Ether',
	'Word-To-Vec_LSTM_Leaking-Ether-to-arbitrary-address',
	'Word-To-Vec_LSTM_Outdated-Solidity-version',
	'Word-To-Vec_LSTM_Timestamp-dependence',
	'Word-To-Vec_LSTM_Ugradeable-contract',
	'Word-To-Vec_LSTM_Unchecked-call-return-value',
	'Word-To-Vec_LSTM_Unprotected-Suicide',
]

def read_data(vul_ratio=0.1, no_vul_ratio=0.1, title=''):
    def format_bytecode(bytecode):
        bytecode = bytecode[:3*MAX_FEATURES].strip()
        if len(bytecode) == 3*MAX_FEATURES-1:
            return bytecode
        return bytecode + ' 00' * (MAX_FEATURES - (len(bytecode)+1)//3)

    df_vul = pd.read_csv('/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/' + title + '.csv', usecols=['BYTECODE']).dropna().drop_duplicates(subset=['BYTECODE']).sample(frac=vul_ratio)
    df_vul['BYTECODE'] = df_vul['BYTECODE'].apply(format_bytecode)
    df_vul['LABEL'] = 1

    df_no_vul = pd.read_csv(NO_VUL_DIR, usecols=['OPCODE']).dropna().drop_duplicates(subset=['OPCODE']).sample(frac=no_vul_ratio)
    df_no_vul.rename(columns={'OPCODE':'BYTECODE'}, inplace=True)
    df_no_vul['BYTECODE'] = df_no_vul['BYTECODE'].apply(format_bytecode)
    df_no_vul['LABEL'] = 0

    df = pd.concat([df_no_vul, df_vul]).sample(frac=1)

    X = df['BYTECODE'].values
    y = df['LABEL'].values

    return X, y

def divide_data(X, y):
    def split_data(data, ratio):
        index = int(len(data) * ratio)
        return data[:index], data[index:]
    
    _, X_test = split_data(X, 0.2)
    _, y_test = split_data(y, 0.2)

    return (X_test, y_test)

    
# Shape: [-1, MAX_FEATURES, 1]
def oneHotEncoding_module(X):
	X = np.array([np.array(x.split()) for x in X])
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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model
    
def run(X, y, title):
	print(title)
	
	input_shape = (X.shape[1], X.shape[2])
	
	(X_test, y_test) = divide_data(X, y)
	model = LSTM_model(input_shape)
	model.load_weights('./' + title + '/model.h5')

	y_pred = model.predict(X_test)
	y_pred = np.array(y_pred, dtype=int)

	result = classification_report(y_test, y_pred, output_dict=True)
	pd.DataFrame(result).transpose().to_csv('./' + title + '/result.csv')

if __name__ == '__main__':
	for index, vul in enumerate(vuls):
		X_origin, y = read_data(vul_ratio=1, no_vul_ratio=1, title=vul)
		
		# 1-hot Encoding
		X = oneHotEncoding_module(X_origin)
		run(X, y, titles[index])
		
		# BOW
		X = bagOfWord_module(X_origin)
		run(X, y, titles[index + 9])
		
		# TF-IDF
		X = tfidf_module(X_origin)
		run(X, y, titles[index + 18])
		
		# Word2Vec
		X = word2vec_module(X_origin)
		run(X, y, titles[index + 27])
		
	
	
