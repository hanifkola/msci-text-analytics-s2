import json


def readdata(filename):
    with open(filename, 'r') as file:
        line = file.readlines()
        return line


def main():
    vocab = {}
    pos_data = readdata('pos.txt')
    neg_data = readdata('neg.txt')
    all_line = pos_data+neg_data
    for idx,line in enumerate(all_line):
        sentence = line.strip().split()
        for word in sentence:
            word = word.lower()
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

    with open('data/vocab.json', 'w') as f:
            json.dump(vocab, f)

if __name__ == '__main__':
    main()