from train.forex_divide_input_output import ForexDivideInputOutput
from train.input_average import InputAverage




def main(_):
    INPUT_SIZE = 60;
    OUTPUT_SIZE = 15;
    test = ForexDivideInputOutput(INPUT_SIZE,OUTPUT_SIZE,"/home/shohdi/projects/learning-tensorflow/projects/deep_learn_finance/conv_4_layer/input/myOldData.csv")
    #print("history : " , test.history , " future : " , test.future )
    inputTuble,outputTuble,mainArr = test.getInputOutput()
    iAvg = InputAverage();
    average = iAvg.getInputAverage(OUTPUT_SIZE,inputTuble,mainArr);

    #print("input ",inputTuble," output ",outputTuble)

main('')