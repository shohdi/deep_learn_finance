from train.forex_fix import ForexFix





def main(_):
    test = ForexFix(60,15)
    print("history : " , test.history , " future : " , test.future )



main('')