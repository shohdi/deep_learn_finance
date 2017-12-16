from train.forex_divide_input_output import ForexDivideInputOutput
from train.input_average import InputAverage

#1/1 - 1/50 - 1/100 - 1/500 - 1/1000


def main(_):
    INPUT_SIZE = 60;
    OUTPUT_SIZE = 15;
    FILE_NAMES = "/home/shohdi/projects/learning-tensorflow/projects/deep_learn_finance/conv_4_layer/input/myOldData.csv";
    test = ForexDivideInputOutput(INPUT_SIZE,OUTPUT_SIZE,FILE_NAMES);
    #print("history : " , test.history , " future : " , test.future )
    inputTuble,outputTuble,mainArr = test.getInputOutput();
    iAvg = InputAverage();
    average = iAvg.getInputAverage(OUTPUT_SIZE,inputTuble,mainArr);
    print('average ' , average);
    #print("input ",inputTuble," output ",outputTuble)

main('')