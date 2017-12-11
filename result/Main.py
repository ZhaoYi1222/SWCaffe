import os
import re

#define file path
filePath = os.getcwd()
fileList = []

#define line match for forward && backward
forwardLine = r"^.+layer\d+ (.+) forward cost time: (.+)$"
backwardLine = r"^.+layer\d+ (.+) Backward cost time: (.+)$"

def init(filePath):
    recentFileList = os.listdir(filePath)
    return recentFileList

if __name__ == '__main__' :
    forwardResults = []
    backwardResults = []
    fileList = init(filePath)
    for file in fileList :
        wholeFileName = filePath + "/" + file
        suffix = ".*log$"
        if not re.match(suffix, file):
            continue
        readFile = open(wholeFileName, "r")
        while True:
            ln = readFile.readline()
            if not ln:
                break
            forwardMatch = re.match(forwardLine, ln)
            if forwardMatch:
                forwardResult = [forwardMatch.group(1), forwardMatch.group(2)]
                forwardResults.append(forwardResult)
            backwardMatch = re.match(backwardLine, ln)
            if backwardMatch:
                backwardResult = [backwardMatch.group(1), backwardMatch.group(2)]
                backwardResults.append(backwardResult)
        backwardResults.reverse()
        #print forwardResults
        #print backwardResults
        if not os.path.exists(filePath + "/result") :
            os.mkdir(filePath + "/result")
        forwardFile = filePath + "/result/" + file.split('.')[0] + "Forward.txt"
        backwardFile = filePath + "/result/" + file.split('.')[0] + "Backward.txt"
        writeForwardFile = open(forwardFile, "w+")
        for forwardResult in forwardResults :
            writeForwardFile.write(forwardResult[0].ljust(80,' ') + " " + forwardResult[1] + '\n')
        writeForwardFile.close()

        writeBackwardFile = open(backwardFile, "w+")
        for backwardResult in backwardResults :
            writeBackwardFile.write(backwardResult[0].ljust(80,' ') + " " + backwardResult[1] + '\n')
        writeBackwardFile.close()
        readFile.close()
