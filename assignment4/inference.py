import sys
from tensorflow import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config import config
import os

def main(data_dir, modeltype):
    with open(os.path.join(data_dir, 'a.txt')) as f:
        text = f.readlines()
    with open(os.path.join(data_dir,'x_train.txt')) as f:
        x_train= f.readlines()
    t = Tokenizer(num_words=config['max_vocab_size'])
    t.fit_on_texts(x_train)
    model = keras.models.load_model(os.path.join(data_dir,"nn_"+modeltype+".model"))
    for lines in text:
        encoded_sent = t.texts_to_sequences(lines)
        padded_sent = pad_sequences(encoded_sent,maxlen = config['max_seq_len'],padding='post')
        result = model.predict(padded_sent)[1]
        if result == 1:
            sentiment ='Positive'
        else: sentiment = 'Negative'
        print (lines, ":",sentiment)



if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
