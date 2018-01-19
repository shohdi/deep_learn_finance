import os as os



class JoinInput:


    def joinInput(self,inputFolder,inputNames):
        files = inputNames.split(";");
        strRet = '';
        for i in range(len(files)):
            oneFile = files[i];
            oneFile = os.path.join(inputFolder,oneFile);
            strRet = strRet + oneFile + ';';
        
        if(strRet != ''):
            strRet = strRet[0:(len(strRet)-1)];
        
        return strRet;
