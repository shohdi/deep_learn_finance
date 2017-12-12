from train.forex_divide_input_output import ForexDivideInputOutput





def main(_):
    test = ForexDivideInputOutput(60,15)
    print("history : " , test.history , " future : " , test.future )
    test.getInputOutput()


main('')