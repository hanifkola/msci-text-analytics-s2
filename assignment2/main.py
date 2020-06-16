import pprint
import pickle
import sys
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def read_csv(data_path):
    with open(data_path) as file:
        data = file.readlines()
    return[' '.join(line.strip().split(',')) for line in data]


def loaddata(data_dir):
    x_train_with_sw = read_csv(os.path.join(data_dir, 'outwsw_train.csv'))
    x_train_without_sw = read_csv(os.path.join(data_dir, 'outwosw_train.csv'))
    x_valid_with_sw = read_csv(os.path.join(data_dir, 'outwsw_valid.csv'))
    x_valid_without_sw = read_csv(os.path.join(data_dir, 'outwosw_valid.csv'))
    x_test_with_sw = read_csv(os.path.join(data_dir, 'outwsw_test.csv'))
    x_test_without_sw = read_csv(os.path.join(data_dir, 'outwosw_test.csv'))
    return x_train_with_sw, x_train_without_sw, x_valid_with_sw, x_valid_without_sw, x_test_with_sw, x_test_without_sw


def loadlabels(data_dir):             #since in my
    with open(os.path.join(data_dir,'labels.csv')) as file:
        data = file.readlines()
    labels = [int(label) for label in data]
    y_train = [1]*labels[0]+[0]*(labels[1]-labels[0]-1)
    y_test = [1]*labels[2]+[0]*(labels[3]-labels[2]-1)
    y_val = [1]*labels[4]+[0]*(labels[5]-labels[4]-1)
    return y_train, y_val, y_test


def train(x_val, y_val,ng1,ng2, picklefile):  #this functon is written with help of TA's code
    print("calling CountVectorizer...")
    count_vec = CountVectorizer(analyzer='word', ngram_range=(ng1, ng2))
    print("transforming vectors with fit function...")
    x_val_count = count_vec.fit_transform(x_val)
    print("calling TfidfTransformer...")
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    print("normalizing the input vector with tf-idf transformer...")
    x_train_tfidf = tfidf_transformer.fit_transform(x_val_count)
    print("calling MultinomialNB...")
    clf = MultinomialNB()
    print("developing the classifier...")
    clf.fit(x_train_tfidf, y_val)
    print('storig trained model as pickle file:',picklefile)
    with open(picklefile, 'wb') as f:
        pickle.dump(clf,f)
    return clf, count_vec, tfidf_transformer


def evaluate(x, y, clf, count_vec, tfidf_trans):    #this functon is written with help of TA's code
    x_count = count_vec.transform(x)
    x_tfidf = tfidf_trans.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        }



def main(data_dir):
    print("loading labels...")
    y_train, y_val, y_test = loadlabels(data_dir)
    print("loading data...")
    x_train_with_sw, x_train_without_sw, x_valid_with_sw, x_valid_without_sw, x_test_with_sw, x_test_without_sw = loaddata(data_dir)
    print("developing and training the model with stop words for unigrams:")
    clf_with_sw11, count_vec_with_sw11, tfidf_transformer_with_sw11 = train(x_train_with_sw, y_train, 1, 1,
                                                                            os.path.join(data_dir, 'mnb_uni.pkl'))
    print("developing and training the model with stop words for bigrams:")
    clf_with_sw22, count_vec_with_sw22, tfidf_transformer_with_sw22 = train(x_train_with_sw, y_train, 2, 2,
                                                                            os.path.join(data_dir, 'mnb_bi.pkl'))
    print("developing and training the model with stop words for unigram+ bigrams:")
    clf_with_sw12, count_vec_with_sw12, tfidf_transformer_with_sw12 = train(x_train_with_sw, y_train, 1, 2,
                                                                    os.path.join(data_dir, 'mnb_uni_bi.pkl'))

    print("developing and training the model without stop words for unigrams:")
    clf_without_sw11, count_vec_without_sw11, tfidf_transformer_without_sw11 = train(x_train_without_sw, y_train,1 ,1 ,
                                                                                     os.path.join(data_dir, 'mnb_uni_ns.pkl'))
    print("developing and training the model without stop words for bigrams:")
    clf_without_sw22, count_vec_without_sw22, tfidf_transformer_without_sw22 = train(x_train_without_sw, y_train, 2, 2,
                                                                               os.path.join(data_dir, 'mnb_bi_ns.pkl'))
    print("developing and training the model without stop words for unigrams:")
    clf_without_sw12, count_vec_without_sw12, tfidf_transformer_without_sw12 = train(x_train_without_sw, y_train, 1, 2,
                                                                               os.path.join(data_dir, 'mnb_uni_bi_ns.pkl'))
    score_with_sw11 = {}
    score_with_sw22 = {}
    score_with_sw12 = {}
    score_without_sw11 = {}
    score_without_sw22 = {}
    score_without_sw12 = {}

    score_with_sw11['val'] = evaluate(x_valid_with_sw, y_val, clf_with_sw11, count_vec_with_sw11, tfidf_transformer_with_sw11)
    score_with_sw22['val'] = evaluate(x_valid_with_sw, y_val, clf_with_sw22, count_vec_with_sw22,
                                      tfidf_transformer_with_sw22)
    score_with_sw12['val'] = evaluate(x_valid_with_sw, y_val, clf_with_sw12, count_vec_with_sw12,
                                     tfidf_transformer_with_sw12)
    score_without_sw11['val'] = evaluate(x_valid_without_sw,y_val, clf_without_sw11, count_vec_without_sw11, tfidf_transformer_without_sw11)
    score_without_sw22['val'] = evaluate(x_valid_without_sw, y_val, clf_without_sw22, count_vec_without_sw22,
                                         tfidf_transformer_without_sw22)
    score_without_sw12['val'] = evaluate(x_valid_without_sw, y_val, clf_without_sw12, count_vec_without_sw12,
                                         tfidf_transformer_without_sw12)

    score_with_sw11['test'] = evaluate(x_test_with_sw, y_test, clf_with_sw11, count_vec_with_sw11, tfidf_transformer_with_sw11)
    score_with_sw22['test'] = evaluate(x_test_with_sw, y_test, clf_with_sw22, count_vec_with_sw22,
                                       tfidf_transformer_with_sw22)
    score_with_sw12['test'] = evaluate(x_test_with_sw, y_test, clf_with_sw12, count_vec_with_sw12,
                                      tfidf_transformer_with_sw12)
    score_without_sw11['test'] = evaluate(x_test_without_sw, y_test, clf_without_sw11, count_vec_without_sw11,
                                       tfidf_transformer_without_sw11)
    score_without_sw22['test'] = evaluate(x_test_without_sw, y_test, clf_without_sw22, count_vec_without_sw22,
                                          tfidf_transformer_without_sw22)
    score_without_sw12['test'] = evaluate(x_test_without_sw, y_test, clf_without_sw12, count_vec_without_sw12,
                                          tfidf_transformer_without_sw12)
    pp = pprint.PrettyPrinter(indent=2)
    print("prediction scores of model trained with data with stopwords for unigrams:")
    pp.pprint(score_with_sw11)
    print("prediction scores of model trained with data with stopwords for bigrams:")
    pp.pprint(score_with_sw22)
    print("prediction scores of model trained with data with stopwords for unigrams and bigrams:")
    pp.pprint(score_with_sw12)
    print("prediction scores of model trained with data without stopwords for unigrams:")
    pp.pprint(score_without_sw11)
    print("prediction scores of model trained with data without stopwords for bigrams:")
    pp.pprint(score_without_sw22)
    print("prediction scores of model trained with data without stopwords for unigrams and bigrams:")
    pp.pprint(score_without_sw12)


if __name__ == '__main__':

    main((sys.argv[1]))