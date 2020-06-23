
#import gensim.downloader as api
import gensim
import os
import pprint
import sys

def main(data_dir):
    print('loading the Word2Vec model...')
    Wvec = gensim.models.Word2Vec.load(os.path.join(data_dir, 'W2v.model'))

    with open (os.path.join(data_dir, 'a.txt')) as f:
        words = f.readlines()
    words = [word.strip().split() for word in words]
    pp = pprint.PrettyPrinter(indent=2)
    for word in words:
        result = Wvec.most_similar(word,topn = 20)
        print('Top simialr words to:',word)
        pp.pprint(result)





if __name__ == '__main__':
    main(sys.argv[1])