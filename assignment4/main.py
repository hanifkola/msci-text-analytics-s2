import numpy as np
from config import config
#from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
import gensim
import os
import sys
# import pandas as pd
# import random
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    """
    load po.txt and neg.txt, create labels and split test sets
    """
    with open(os.path.join(data_dir, 'pos.txt')) as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, 'neg.txt')) as f:
        neg = f.readlines()
    all_line = pos+neg
    labels =[1]*len(pos)+[0]*len(neg)
    x_train,x_test,y_train,y_test = train_test_split(all_line,labels,test_size=0.1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train,x_test,y_train,y_test




def main(data_dir):
    print('loading text data...')
    x_train, x_test, y_train, y_test = load_data(data_dir)
    with open(os.path.join(data_dir,'x_train.txt'),'w') as f:
        f.write(str(x_train))
    print('text preprocessing...')
    t = Tokenizer(num_words = config['max_vocab_size'])
    t.fit_on_texts(x_train)
    vocab_size = len(t.word_index) +1
    encoded_x_train = t.texts_to_sequences(x_train)
    encoded_x_test =t.texts_to_sequences(x_test)
    #print(encoded_x_train[1:10])
    padded_x_train = pad_sequences(encoded_x_train,maxlen = config['max_seq_len'],padding='post')
    padded_x_test = pad_sequences(encoded_x_test,maxlen = config['max_seq_len'],padding = 'post')
    #print(padded_x_train[1:10])
    #print(y_train[1:10])
    print('importing the pre-trained Word2vecs...')
    W2vec= gensim.models.Word2Vec.load(os.path.join(data_dir, 'W2v.model'))
    print('developing embedding matrix with Word2veccs...')
    embeddings_matrix = np.random.uniform(-.05,0.05,
                                          size= (vocab_size,
                                                 config['embedding_dim']))
    for word,i in t.word_index.items():
        try:
            embeddings_vector = W2vec[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

    print('creating the model...')
    model = Sequential()
    e= Embedding(vocab_size,config['embedding_dim'],weights=[embeddings_matrix],
                 input_length=config['max_seq_len'],trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(32,activation=config['modeltype']))
    model.add(Dropout(config['Dropout']))
    #model.add(LSTM(32, dropout=config['Dropout'], recurrent_dropout=0.5))
    model.add(Dense(1,kernel_regularizer=l2(config['l2regularize']),activation='softmax'))
    print('compile the model...')
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    print(model.summary())
    print('Training the model with train data...')
    model.fit(padded_x_train,y_train,epochs=config['epochs'],verbose=1)
    print('Evaluate the model with test data...')
    loss,accuracy = model.evaluate(padded_x_test,y_test,verbose=1)
    print('the accuracy of the model on test data is:%f' %(accuracy*100),'%')
    model.save(os.path.join(data_dir,'nn_'+config['modeltype']+'.model'))
if __name__ == '__main__':
    main(sys.argv[1])



