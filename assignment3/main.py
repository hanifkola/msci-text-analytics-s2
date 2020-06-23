

import sys
import os
from gensim.models import Word2Vec


def main(data_dir = 'data'):
    with open(os.path.join(data_dir,'pos.txt')) as f:
        pos_lines = f.readlines()
    with open(os.path.join(data_dir,'neg.txt')) as f:
        neg_lines = f.readlines()
    all_lines = pos_lines + neg_lines
    print('spliting the lines to words')
    all_lines = [line.strip().split() for line in all_lines]
    print('training the Word2Vec model by gensim...')
    W2v = Word2Vec(all_lines,size = 200, window = 5, min_count=5, workers=3)
    W2v.save(os.path.join(data_dir,'W2v.model'))


if __name__ == '__main__':
    main(sys.argv[1])