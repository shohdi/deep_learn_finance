import sys
import os
import codecs
import re
from train.read_date_data import ReadDateFile



def fix_data():
    print('start fixing')
    data = ReadDateFile()
    listRead = list()
    listRead.append(os.path.join('input','trainYear1.csv'))
    listRead.append(os.path.join('input','trainYear2.csv'))
    listRead.append(os.path.join('input','trainYear3.csv'))
    readFileName = ';'.join(listRead)
    print('read files ',readFileName)

    writeFileName = os.path.join('input','trainData.csv')
    data.readFile(readFileName)
    #print('images ',data.myDataImages)
    #print('labels ',data.myDataLabels)

    with open(writeFileName, 'w') as f:
        imagesLength = len(data.myDataImages)
        for i in range(imagesLength):
            oneImage = data.myDataImages[i]
            oneLabel = data.myDataLabels[i]
            oneImageLen = len(oneImage)
            oneLabelLen = len(oneLabel)
            for j in range(oneImageLen):
                val = oneImage[j]
                f.write(str( val))
                f.write(',')
                #if(j< (oneImageLen-1)):
                #    f.write(',')
            
            for j in range(oneLabelLen):
                val = oneLabel[j]
                f.write(str( val))
                if(j< (oneLabelLen-1)):
                    f.write(',')
            
            if(i<imagesLength-1):
                f.write('\n')

    return True








def main(*args):
    print(args)
    fix_data()

if __name__ == '__main__':
    main(*sys.argv[1:])