import json
import pickle
import sys
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest,chi2


def main( input= 'test.txt',clstype = 'mnb_uni'):
    data_dir = "data"
    with open(os.path.join(data_dir,'vocab.json')) as file:
        vocab = json.load(file)
    if clstype ==  'mnb_uni' or 'mnb_uni_ns' :
        ng1= ng2 = 1
    elif clstype == 'mnb_bi' or 'mnb_bi_ns':
        ng1 = ng2 = 2
    else:
        ng1 = 1
        ng2 = 2

    modelfile = os.path.join(data_dir,clstype+".pkl")
    with open(modelfile, 'rb') as file:
        clfmodel = pickle.load(file)
    with open(input, 'r') as f:
        sentences = f.readlines()

    for lines in sentences:
        words = lines.strip().split(' ')
        count_vec = CountVectorizer(ngram_range=(ng1, ng2),vocabulary=list(set(vocab)))

        tfidf_transformer = TfidfTransformer()
        x_count = count_vec.fit_transform(words)
        #x_count_fit = count_vec.fit(x_count)
        x_tfidf = tfidf_transformer.fit_transform(x_count)

        #x_train_count = tfidf_transformer.transform(x_count)
        #selector = SelectKBest(chi2,k=10)
        x_tfidf = tfidf_transformer.transform(x_tfidf)

        #x_tfidf_fit = selector.fit(x_tfidf,words)
        #clfmodel.fit(selector.transform(x_tfidf),words)
        #features_test_cv = selector.transform(tfidf_transformer.transform(count_vec.transform(x_count)))

        preds = clfmodel.predict(x_tfidf)
        print(preds)


if __name__ == '__main__':

    #user_arg = sys.argv()
    main()