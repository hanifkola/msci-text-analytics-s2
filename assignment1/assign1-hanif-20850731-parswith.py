
# this function would tokenize input file and write it in output file with removing stop words.import csv
def  parswith(inputfile,outputfile):
    #put stopwords in a list
    stopworddata = []
    with open("stopwords.txt", "r") as f:
        for line in f:
            stopworddata = stopworddata + line.split(',')

    output = open(outputfile, "w")
    with open(inputfile, 'r') as file:
        # reading each line
        for line in file:
            # reading each word
            for word in line.split():
                #check if the word is stop word or not
                word = word.lower()
                check = 0
                for stopword in stopworddata:
                    if word == stopword:
                        check = 1
                if check ==0:
                    # writing in a output file
                    if word[-1] == ".":
                        #remove dot from the last word in the sentence
                        word = word[:-1]
                        output.write(word)
                        output.write("\n")
                    else:
                        output.write(word)
                        output.write(",")
                else:
                    continue
    output.close
parswith("neg.txt","out_neg_W.csv")
parswith("pos.txt","out_pos_W.csv")
#parswith("test.txt","test.csv")

