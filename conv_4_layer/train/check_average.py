from train.read_file import ReadFile
import numpy as np



a = ReadFile("input/testYear.csv")
arr =a.readMultiFiles()

print(arr)

nparr =np.array(arr[0:79080])
nparr=np.reshape(nparr,(-1,60))
print(nparr)1

