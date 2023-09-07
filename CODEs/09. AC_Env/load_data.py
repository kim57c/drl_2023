import yfinance as yf
import os


def save(name, df):
    
    path = './09. AC_Env'
    target = path+'/'+name+'.csv' 
    
    if os.path.exists ( target ) :
        os.remove(target)
    
    df.to_csv(target,index=False,header=False)
    
    
    
    
def load(name):
    
    tick = yf.Ticker(name)
    
    dataframe = tick.history(
        # start='2022-11-02', 
        # end='2022-11-30', 
        # period='60d',
        # period='2y',
        period='1y',
        interval='1h', 
        # interval='30m', 
        back_adjust=True, 
        auto_adjust=True, 
        prepost=False)
        
    if dataframe.empty :
        print(f'{name} empty!')
    else :
        save(name, dataframe)
        print(f'{name} saved!')
        
load ('BTC-KRW')
