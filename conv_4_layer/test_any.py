from train.forex_divide_input_output import ForexDivideInputOutput





def main(_):
    test = ForexDivideInputOutput(60,15,"/home/shohdi/projects/learning-tensorflow/projects/deep_learn_finance/conv_4_layer/input/myOldData.csv")
    print("history : " , test.history , " future : " , test.future )
    inputTuble,outputTuble,mainArr = test.getInputOutput()
    print("input ",inputTuble," output ",outputTuble)

main('')