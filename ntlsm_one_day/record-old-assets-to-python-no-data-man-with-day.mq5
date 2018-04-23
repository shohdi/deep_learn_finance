//+------------------------------------------------------------------+
//|                                                  TradeByTick.mq5 |
//|                        Copyright 2017, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"






//+------------------------------------------------------------------+
//| My custom types                                   |
//+------------------------------------------------------------------+

struct MqlCandle
 {
   double Close;
   double Open;
    int Dir;
   double High;
    double Low;
    double Volume;
    datetime Date;
 };
 
 struct MqlSimpleCandle
 {
      double close;
      double open;
      double high;
      double low;
      double calcPrice;
 };
 
 struct MqlCandleTicks
 {
   MqlCandle candle;
   double UpTicks;
   double DownTicks;
   double TicksVolume;
   double DownMoves;
   double UpMoves;
 };
 
 
  
//+------------------------------------------------------------------+
//| variables needed                                   |
//+------------------------------------------------------------------+


input int noOfTradePeriods = 1;
input int noOfSecondsToAlertBefore = 55;
input int startHour = 0;
input int endHour = 23;


MqlCandleTicks liveCandle;
MqlCandleTicks currentCandle;
MqlCandleTicks lastCandle;
MqlCandleTicks beforeLastCandle;
MqlTick currentTick;
MqlTick lastTick;
bool alertedSegnal = false;


int noOfSuccess = 0;
int noOfFail = 0;

double oldData[];







//+------------------------------------------------------------------+
//| My custom functions                                   |
//+------------------------------------------------------------------+




bool calcTime()
{
    datetime currentDate = TimeCurrent();
        
          MqlDateTime strucTime;
          TimeToStruct(currentDate,strucTime);
          
          return (strucTime.hour >= startHour && strucTime.hour <= endHour);
}

int getDay(datetime current)
{
       MqlDateTime strucTime;
          TimeToStruct(current,strucTime);
          
          
          return strucTime.day;
}

int getHour(datetime current)
{
       MqlDateTime strucTime;
          TimeToStruct(current,strucTime);
          
          
          return strucTime.hour;
}

int getMinute(datetime current)
{
       MqlDateTime strucTime;
          TimeToStruct(current,strucTime);
          
          
          return strucTime.min;
}



MqlCandle getCandle (int pos)
 {
      MqlCandle ret;
      
      double closes[1];
      double opens[1];
      double highs[1];
      double lows[1];
      long volumes[1];
       datetime dates[1];
      CopyClose(_Symbol,_Period,pos,1,closes);
       CopyOpen(_Symbol,_Period,pos,1,opens);
      CopyHigh(_Symbol,_Period,pos,1,highs);
      CopyLow(_Symbol,_Period,pos,1,lows);
      CopyTime(_Symbol,_Period,pos,1,dates);
      ret.Volume = 1;
      int volFound = CopyRealVolume(_Symbol,_Period,pos,1,volumes);
      if(volFound > 0)
      {
         if(volumes[0] > 0)
         {
               ret.Volume = volumes[0];
         }
      
      }
      ret.Date = dates[0];
      ret.Close = closes[0];
      ret.Open = opens[0];
      ret.High = highs[0];
      ret.Low = lows[0];
      if(ret.Open < ret.Close)
         ret.Dir = 1;
     else if (ret.Open > ret.Close)
         ret.Dir = -1;
     else
         ret.Dir = 0;
         
         
         return ret;
         
            
 }
 
 
 string PrintCandle(MqlCandleTicks &candleTicks)
 {
      string dirStr = "equal";
      if(candleTicks.candle.Dir > 0)
      {
         dirStr = "green";
      }
      else if (candleTicks.candle.Dir < 0)
      {
         dirStr = "red";
      }
      return 
               "candle date : " + candleTicks.candle.Date + "  "
               + "direction : " + dirStr + "  "
                + "open : " + candleTicks.candle.Open + "  "
                 + "close : " + candleTicks.candle.Close + "  "
                  + "high : " + candleTicks.candle.High + "  "
                  + "low : " + candleTicks.candle.Low + "  "
                 
                 
               + "up ticks : " + candleTicks.UpTicks + "  "
                + "down ticks : " + candleTicks.DownTicks + "  "
                 + "ticks volume : " + candleTicks.TicksVolume  + " "
                  + "down moves : " + candleTicks.DownMoves + "  "
                 + "up  moves : " + candleTicks.UpMoves ;
                 
 }
 
void saveTickToCandle (MqlCandleTicks &candleTicks,MqlTick &lastTick , MqlTick &currentTick)
{
       
         double move = 0;
         //add tick information to currentCandle
         if(currentTick.bid > lastTick.bid)
         {
           
            candleTicks.UpTicks = candleTicks.UpTicks +1;
            move = currentTick.bid - lastTick.bid;
            candleTicks.UpMoves = candleTicks.UpMoves + move;
         }
         else if(currentTick.bid < lastTick.bid)
         {
           
            candleTicks.DownTicks  = candleTicks.DownTicks + 1;
            move = lastTick.bid - currentTick.bid;
              candleTicks.DownMoves  = candleTicks.DownMoves + move;
            
         }
         
         
         
         candleTicks.TicksVolume = candleTicks.TicksVolume +  currentTick.volume;
         

}

double compareCandles (MqlCandle &old,MqlCandle &newC)
{
      if (newC.High > old.High
      && newC.Close > old.Close
      && newC.Low > old.Low)
      {
         return 1;
      }
      else if (newC.High < old.High
      && newC.Close < old.Close
      && newC.Low < old.Low)
      {
         return -1;
      }
      else
      {
         return 0;
      }
      
}


double getDirectionOfNoOfPeriods (int pos,int noOfPeriods)
{
      MqlCandle lastCandle = getCandle(pos);
      MqlCandle startCandle = getCandle(pos+(noOfPeriods));
      if(startCandle.Close > lastCandle.Close)
      {
         return -1;
      }
      else if (startCandle.Close < lastCandle.Close)
      {
         return 1;
      }
      else
      {
         return 0;
      }
}





//+------------------------------------------------------------------+
//|  indicator functions                                   |
//+------------------------------------------------------------------+

/*
      if ( beforeCandle.Close > liveCandle.Close
            && beforeBeforeCandle.Close > beforeCandle.Close
            &&  compareCandles(oldCandle,beforeBeforeCandle) > 0 
             &&  compareCandles(veryOldCandle,oldCandle) > 0
              &&  compareCandles(beforeVeryOldCandle,veryOldCandle) > 0
       )
       {
            return 1;
       }
       else if (beforeCandle.Close < liveCandle.Close
            && beforeBeforeCandle.Close < beforeCandle.Close
            &&  compareCandles(oldCandle,beforeBeforeCandle) < 0 
             &&  compareCandles(veryOldCandle,oldCandle) <0
             &&  compareCandles(beforeVeryOldCandle,veryOldCandle) < 0
       )
       {
            return -1;
       }
       else
      {
         return 0;
      }       

*/

void normalizeCandlesToMinimumWork(MqlCandle &myCalcCandles[],MqlSimpleCandle &ret[])
{
   double base = 9999999;
   for(int i=0;i< 5;i++)
   {
        if(myCalcCandles[i].Low < base)
        {
            base = myCalcCandles[i].Low;
        }
   }
   
   double mostHigh = 0;
     for(int i=0;i< 5;i++)
      {
           if(myCalcCandles[i].High > mostHigh)
           {
               mostHigh = myCalcCandles[i].High;
           }
      }
      
      ArrayResize(ret,6);
      for (int i=0;i<5;i++)
      {
            ret[i].calcPrice = base;
            ret[i].close = normalizeOneDoubleValue(base,mostHigh,myCalcCandles[i].Close);
              ret[i].open = normalizeOneDoubleValue(base,mostHigh,myCalcCandles[i].Open);
              ret[i].high = normalizeOneDoubleValue(base,mostHigh,myCalcCandles[i].High);
              ret[i].low = normalizeOneDoubleValue(base,mostHigh,myCalcCandles[i].Low);
              
      }
      
      
      datetime lastCandleDate = myCalcCandles[4].Date;
      
      
      ret[5].open = getDay(lastCandleDate);
      ret[5].close = getHour(lastCandleDate);
      ret[5].high = getMinute(lastCandleDate); 
      
        
      
      
   
}

double normalizeOneDoubleValue (double base , double maxHigh,double value)
{
   if((maxHigh - base) == 0)
   {
         return 0;
   }
    double inValue = value-base;
    double percValue = (inValue/(maxHigh-base)) * 100;
    percValue = MathRound(percValue);
    return percValue;
}


void PrintSimpleCandles (MqlSimpleCandle & ret[])
{
      int size = ArraySize(ret);
      for(int i=0;i<size;i++)
      {
            Print(" candle no : " + i + " : " +
            " calc-price : " + ret[i].calcPrice +
            " close : " + ret[i].close + 
            " open : " + ret[i].open +
            " high : " + ret[i].high +
             " low : " + ret[i].low);
            
      }
}











void shohdiCalculateSuccessFail ()
{
        //if(!calcTime())
        //    return;
        MqlCandle lastCreated = getCandle(1);
       
         int size = ArraySize(oldData);
         int newSize = size + 6;
         ArrayResize(oldData,newSize);
         oldData[size] =  getDay(lastCreated.Date);
         oldData[size+1] =  getHour(lastCreated.Date);
         oldData[size+2] = lastCreated.Open;
         oldData[size+3] = lastCreated.Close;
         oldData[size+4] = lastCreated.High;
         oldData[size+5] = lastCreated.Low;        
        /* Print(
         "old data open : " + oldData[size] +
           "old data close : " + oldData[size + 1] +
             "old data high : " + oldData[size + 2] +
              "old data low : " + oldData[size + 3] );*/
      
}





//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
  
      
  
//--- create timer
   EventSetTimer(1);
      
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  
      int size = (ArraySize(oldData));
      
      int filehandle=FileOpen("myOldData.csv",FILE_WRITE|FILE_CSV);
      // FileWrite(filehandle, size);
       for (int i=0;i<size;i++)
       {
           
                  FileWrite(filehandle,oldData[i] );
                  
                 
           
       }
      
      
       FileClose(filehandle);
       Print("File ok!");
       
      
//--- destroy timer
   EventKillTimer();
      
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- 
         SymbolInfoTick(_Symbol,currentTick);
         
         
         
         MqlCandleTicks lastTickCandle = liveCandle;
         liveCandle.candle = getCandle(0);
       
         
         if(liveCandle.candle.Date != lastTickCandle.candle.Date)
         {
         
                  //before last candle becomes last candle
                  beforeLastCandle = lastCandle;
                //last candle becomed current
                lastCandle = currentCandle;
                //new candle become closed
               currentCandle = lastTickCandle;
              currentCandle.candle = getCandle(1);
         
               //clear live candle
               liveCandle.DownTicks  = 0;
               liveCandle.UpTicks = 0;
               liveCandle.TicksVolume = 0;
               
               
               //clear any new candle variables
               alertedSegnal = false;
               
              
               
             
               
               
               //do any new candle operations
               shohdiCalculateSuccessFail();
               
               //print result if found
               
               
         }
         
         
        
         
         //save tick data to use in any calc
         saveTickToCandle(liveCandle,lastTick,currentTick);
         
        
         
         //move tick
         lastTick = currentTick;
        
         
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
  
  return ;
  

   
          
          
   
  }
//+------------------------------------------------------------------+
