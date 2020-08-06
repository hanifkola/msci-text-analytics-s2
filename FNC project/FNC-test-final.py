from tensorflow import keras
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
from collections import Counter
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

max_len_head = 20
max_len_body = 80

def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count




nltk.download('stopwords')

stop = set(stopwords.words("english"))


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)




def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)



def main():
    print('importing data...')
    headers_train = pd.read_csv('train_stances.csv')
    bodies_train = pd.read_csv('train_bodies.csv')
    headers_test = pd.read_csv('competition_test_stances.csv')
    bodies_test = pd.read_csv('test_bodies.csv')

    print('cleaning data...')
    headers_test["Headline"] = headers_test.Headline.map(lambda x: remove_punct(x))
    headers_train["Headline"] = headers_train.Headline.map(lambda x: remove_punct(x))
    bodies_train["articleBody"] = bodies_train.articleBody.map(lambda x: remove_punct(x))
    bodies_test["articleBody"] = bodies_test.articleBody.map(lambda x: remove_punct(x))

    headers_test["Headline"] = headers_test["Headline"].map(remove_stopwords)
    headers_train["Headline"] = headers_train["Headline"].map(remove_stopwords)
    bodies_train["articleBody"] = bodies_train["articleBody"].map(remove_stopwords)
    bodies_test["articleBody"] = bodies_test["articleBody"].map(remove_stopwords)

    print("Tokenizing...")
    text = headers_train.Headline.append(bodies_train.articleBody)
    counter = counter_word(text)
    num_words = len(counter)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(headers_train.Headline.append(bodies_train.articleBody))


    print('producing test data...')

    test_sequences_header = tokenizer.texts_to_sequences(headers_test.Headline)
    test_sequences_body = tokenizer.texts_to_sequences(bodies_test.articleBody)

    test_padded_head = pad_sequences(
        test_sequences_header, maxlen=max_len_head, padding="post", truncating="post"
    )
    test_padded_body = pad_sequences(
        test_sequences_body, maxlen=max_len_body, padding="post", truncating="post"
    )
    test_padded = np.zeros((len(headers_test), max_len_head + max_len_body))
    for i in tqdm(range(len(headers_test))):
        BodyID = headers_test["Body ID"][i]
        j = bodies_test[bodies_test["Body ID"] == BodyID].index
        test_padded[i] = np.append(test_padded_head[i], test_padded_body[j])

    print('loading models...')

    model1 = keras.models.load_model("model400-1.model")
    headers_test["prediction"] = ''

    print('predicting the output...')
    for i in tqdm(range(len(headers_test))):
        input = test_padded[i]
        input_t = input.reshape(1, 100)

        result1 = model1.predict(input_t)
        result_no = np.argmax(result1)
        if result_no == 0:
            headers_test["prediction"][i] = 'unrelated'
        if result_no == 1:
            headers_test["prediction"][i] = 'agree'
        if result_no == 2:
            headers_test["prediction"][i] = 'disagree'
        if result_no == 3:
            headers_test["prediction"][i] = 'discuss'
    headers_test.to_csv('model400-1.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()