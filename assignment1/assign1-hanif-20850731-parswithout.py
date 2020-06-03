# this function would tokenize input file and write it in output file without removing stop words.
def  parswithout(inputfile,outputfile):
    output = open(outputfile, "w")
    output.write("[")
    with open(inputfile, 'r') as file:
        # reading each line
        for line in file:
            # reading each word
            for word in line.split():
                # writing in a output file
                if word[-1] == ".":
                    #remove dot from the last word in the sentence
                    word = word[:-1]
                    output.write(word)
                    output.write("]")
                    output.write("\n")
                    output.write("[")
                else:
                    output.write(word)
                    output.write(",")
    output.close
parswithout("neg.txt","out_neg_WO.csv")
parswithout("pos.txt","out_pos_WO.csv")



