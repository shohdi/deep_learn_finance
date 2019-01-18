import os
import csv
import glob
import numpy as np
import collections

#<high>,<low>,<open>,<close>,<avgm>,<avgh>,<avgd>,<month>,<dayofmonth>,<dayofweek>,<hour>,<minute>,<ask>,<bid>,<volume>
#'high','low','open','close','avgm','avgh','avgd','month','dayofmonth','dayofweek','hour','minute','ask','bid','volume'
Prices = collections.namedtuple('Prices', field_names=['high','low','open','close','avgm','avgh','avgd','month','dayofmonth','dayofweek','hour','minute','ask','bid','volume'])


def read_csv(file_name, sep=',', filter_data=True, fix_open_price=False):
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<high>' not in h and sep == ',':
            return read_csv(file_name, ';')
        indices = [h.index(s) for s in ('<high>','<low>','<open>','<close>','<avgm>','<avgh>','<avgd>','<month>','<dayofmonth>','<dayofweek>','<hour>','<minute>','<ask>','<bid>')]
        high,low,myopen,close,avgm,avgh,avgd,month,dayofmonth,dayofweek,hour,minute,ask,bid =  [], [], [], [],[],[],[],[],[],[],[],[],[],[]
        count_out = 0
        count_filter = 0
        count_fixed = 0
        prev_vals = None
        for row in reader:
            vals = list(map(float, [row[idx] for idx in indices]))
            if filter_data and any(map(lambda v: v <= 0, vals[0:4])):
                count_filter += 1
                continue

            phigh,plow,popen,pclose,pavgm,pavgh,pavgd,pmonth,pdayofmonth,pdayofweek,phour,pminute,pask,pbid = vals

            # fix open price for current bar to match close price for the previous bar
            if fix_open_price and prev_vals is not None:
                pphigh,pplow,ppopen,ppclose,ppavgm,ppavgh,ppavgd,ppmonth,ppdayofmonth,ppdayofweek,pphour,ppminute,ppask,ppbid = prev_vals
                if abs(popen - ppclose) > 1e-8:
                    count_fixed += 1
                    popen = ppclose
                    plow = min(plow, popen)
                    phigh = max(phigh, popen)
            count_out += 1

            
            high.append(phigh);
            low.append(plow);
            myopen.append(popen);
            close.append(pclose);
            avgm.append(pavgm);
            avgh.append(pavgh);
            avgd.append(pavgd);
            month.append(pmonth);
            dayofmonth.append(pdayofmonth);
            dayofweek.append(pdayofweek);
            hour.append(phour);
            minute.append(pminute);
            ask.append(pask);
            bid.append(pbid);



            prev_vals = vals


    print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
        count_filter + count_out, count_filter, count_fixed))
    
    return Prices(high=np.array(high, dtype=np.float32),
                  low=np.array(low, dtype=np.float32),
                  open=np.array(myopen, dtype=np.float32),
                  close=np.array(close, dtype=np.float32),
                  avgm=np.array(avgm, dtype=np.float32),
                  avgh=np.array(avgh, dtype=np.float32),
                  avgd=np.array(avgd, dtype=np.float32),
                  month=np.array(month, dtype=np.float32),
                  dayofmonth=np.array(dayofmonth, dtype=np.float32),
                  dayofweek=np.array(dayofweek, dtype=np.float32),
                  hour=np.array(hour, dtype=np.float32),
                  minute=np.array(minute, dtype=np.float32),
                  ask=np.array(ask, dtype=np.float32),
                  bid=np.array(bid, dtype=np.float32),
                  volume = None)
                  


def prices_to_relative(prices):
    return prices;


def load_relative(csv_file):
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result
