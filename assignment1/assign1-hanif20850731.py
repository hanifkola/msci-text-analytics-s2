
import random

output1 = open("outwsw.csv", "w")               #tokenize file with stop words
output2 = open("outwosw.csv", "w")              #tokenize file without stop words
output3 = open("outwsw_train.csv","w")          #train set of tokenize file with stop words
output4 = open("outwsw_test.csv","w")           #test set of tokenize file with stop words
output5 = open("outwsw_valid.csv","w")          #validations set of tokenize file with stop words
output6 = open("outwosw_train.csv","w")         #train set of tokenize file without stop words
output7 = open("outwosw_test.csv","w")          #test set of tokenize file without stop words
output8 = open("outwosw_valid.csv","w")         #validations set of tokenize file without stop words

stopworddata = []
with open("stopwords.txt", "r") as f:
    for line in f:
        stopworddata = stopworddata + line.split(',')

with open("pos.txt", 'r') as file:
    pos_line = file.readlines()
with open("neg.txt", 'r') as file:
    neg_line = file.readlines()

all_line = pos_line+neg_line
for line in all_line:
    rand = random.random()
    for word in line.split():
        word = word.lower()
        if word[-1] == ".":
            output1.write(word[0:-1]+"\n")
            if rand < 0.8:
                output3.write(word[0:-1]+"\n")
            elif rand < 0.9:
                output4.write(word[0:-1]+"\n")
            else:
                output5.write(word[0:-1]+"\n")
            if word not in stopworddata:
                output2.write(word[0:-1]+"\n")
                if rand < 0.8:
                    output6.write(word[0:-1] + "\n")
                elif rand < 0.9:
                    output7.write(word[0:-1] + "\n")
                else:
                    output8.write(word[0:-1] + "\n")
        else:
            output1.write(word+",")
            if rand < 0.8:
                output3.write(word+",")
            elif rand < 0.9:
                output4.write(word+",")
            else:
                output5.write(word+",")
            if word not in stopworddata:
                output2.write(word+",")
                if rand < 0.8:
                    output6.write(word + ",")
                elif rand < 0.9:
                    output7.write(word + ",")
                else:
                    output8.write(word + ",")
output1.close()
output2.close()
output3.close()
output4.close()
output5.close()
output6.close()
output7.close()
output8.close()



