import numpy as np 
import pandas as pd
import pickle

class generate_pos():

    def __init__(self):
        pass

    def load_model(file):
        try:
            fp=open(file,'rb')
            result=pickle.load(fp)
            fp.close()
        except Exception as e:
            result = 0
            print(e)
        return result

    def kelly_criteria(a,b,p,q):
        # p: the prediction accuracy of rising 
        # q: the prediction accuracy of jumping 
        # a: the rising pct 
        # b: the jumping pct 
        pos = (p/a) - (q/b)
        return pos 
    # from prediction to trading position 


    def three_label_pos(in_file, out_file, pos):
        '''
        3-label: 
        Trade with constatnt intial weight 
        1. If pred_0 = 1, close until next -1, add pos when 1
        2. If pred_0 = 0, pass
        3. If pred_0 = -1, close until next 1, add pos when -1
        '''
        df = pd.read_csv('./Prediction/'+ in_file +'.csv', header = 0)
        df['cur_pos'] = np.zeros(len(df)) - 1

        for i in range(len(df)):
            if df.iloc[i,1] == 1:
                if i == 0: 
                    df.iloc[i,2] = pos 
                else: 
                    if (df.iloc[i-1,2] > 0):
                        df.iloc[i,2] = df.iloc[i,2] + pos 
                    elif (df.iloc[i-1,2] < 0): 
                        df.iloc[i,2] = 0 
                    else: 
                        df.iloc[i,2] = pos 

            elif df.iloc[i,1] == 0: 
                if i == 0: 
                    df.iloc[i,2] = 0
                else: 
                    df.iloc[i,2] = df.iloc[i-1,2]

            elif df.iloc[i,1] == -1: 
                if i == 0: 
                    df.iloc[i,2] = -pos 
                else: 
                    if (df.iloc[i-1,2] > 0):
                        df.iloc[i,2] = 0
                    elif (df.iloc[i-1,2] < 0): 
                        df.iloc[i,2] = df.iloc[i,2] - pos 
                    else: 
                        df.iloc[i,2] = -pos 
            
            df.to_csv('./Prediction/'+ out_file +'.csv')
        return 
   


    
    def five_label_pos(in_file, out_file, pos_1, pos_2):
        df = pd.read_csv('C:/Users/Jialu/Documents/Code/FYP/Prediction/'+ in_file +'.csv', header = 0)
        df.columns = ['timestamp','Pa1','Pa2','Pb2','Pb1','Va1','prediction']
        df = df[['timestamp','prediction']]

        df['cur_pos'] = np.zeros(len(df)) - 1
        cur_pos = []
        if df.iloc[0,1] == 2:     
            print('2 true')
            df.iloc[0,2] = pos_2
        elif df.iloc[0,1] == 1:
            print('1 true')
            df.iloc[0,2] = pos_1
        elif df.iloc[0,1] == 0: 
            print('0 true')
            df.iloc[0,2] = 0
        elif df.iloc[0,1] == -1: 
            print('-1 true')
            df.iloc[0,2] = -pos_1
        elif df.iloc[0,1] == -2: 
            print('-2 true')
            df.iloc[0,2] = -pos_2
        
        for i in range(1,len(df)):
            if df.iloc[i,1] == 2:
                if (df.iloc[i-1,2] > 0):
                    df.iloc[i,2] = df.iloc[i-1,2] + pos_2
                elif (df.iloc[i-1,2] < 0): 
                    df.iloc[i,2] = 0 
                else: 
                    df.iloc[i,2] = pos_2

            elif df.iloc[i,1] == 1:     
                if (df.iloc[i-1,2] > 0):
                    df.iloc[i,2] = df.iloc[i-1,2] + pos_1
                elif (df.iloc[i-1,2] < 0): 
                    df.iloc[i,2] = 0 
                else: 
                    df.iloc[i,2] = pos_1

            elif df.iloc[i,1] == 0: 
                df.iloc[i,2] = df.iloc[i-1,2]

            elif df.iloc[i,1] == -1:    
                if (df.iloc[i-1,2] > 0):
                    df.iloc[i,2] = 0
                elif (df.iloc[i-1,2] < 0): 
                    df.iloc[i,2] = df.iloc[i-1,2] - pos_1
                else: 
                    df.iloc[i,2] = -pos_1
            
            elif df.iloc[i,1] == -2:  
                if (df.iloc[i-1,2] > 0):
                    df.iloc[i,2] = 0
                elif (df.iloc[i-1,2] < 0): 
                    df.iloc[i,2] = df.iloc[i-1,2] - pos_2
                else: 
                    df.iloc[i,2] = -pos_2
            
            # df = df[['timestamp','cur_pos']]
            df.to_csv('C:/Users/Jialu/Documents/Code/FYP/Prediction/'+ out_file +'.csv')
        return 
   
    
    
   

    