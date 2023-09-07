

import csv
import numpy as np
    
class BtcEnv :
    
    def __init__(self): 

        self.datas = []
        file = './09. AC_Env/BTC-KRW.csv'
        
        with open(file, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Open,High,Low,Close,Volume,Dividends,Stock Splits
            for row in rows :
                data = list(map(float, row[0:4]))
                self.datas.append(data)

        self.distance = 7
        self.seq = self.distance
        
        self.seed_money = 10000
        
        self.cash = self.seed_money
        self.btc = 0

    # 현재 시점의 하이와 로우의 가운데 가격으로 설정 
    # Open,High,Low,Close
    def current_btc_price(self):
        d = self.datas[self.seq]
        return d[2]+(d[1]-d[2])/2
    
    
    def state ( self ):
        s = []
        
        for i in range (self.distance-1):
            data = self.datas[self.seq-1]
            
            for j in range(4) :
                s.append ( self.datas[self.seq-2-i][j]-data[j])
        return s 
    
    
    
    def assets (self):
        return self.cash, self.btc * self.current_btc_price()
        
        
    def reset ( self ):
        
        self.seq = self.distance
        self.cash = self.seed_money
        self.btc = 0
        
        return self.state ()
        
    
    def step ( self, action ) :
        
        cur_cash, cur_btc = self.assets()
        total_asset = cur_cash+cur_btc
        
        cur_btc_price = self.current_btc_price()
        
        if action == 0 :
            self.btc = self.btc + (self.cash/cur_btc_price)
            self.cash = 0
            
        else :
            self.cash = self.cash + (self.btc*cur_btc_price)
            self.btc = 0
            
        self.seq = self.seq+1 
        
        next_cash, next_btc = self.assets()
        
        # 다음 상태
        s_prime = self.state ()
        
        # 보상
        r = (next_cash+next_btc)-total_asset
        
        # 데이터의 마지막이면 종료 반환
        done = self.seq == len(self.datas)-1
        
        return s_prime, r, done
        
    
    


if __name__ == '__main__':
    
    btcenv = BtcEnv()
    s = btcenv.reset()
    
    actions = [0.1,0.9]
    done = False
    while done == False:
        s_prime, r, done = btcenv.step(actions)