import numpy as np 
import pandas as pd

class generate_pos():

    def __init__(self):
        pass

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
        df = pd.read_csv('./Prediction/'+ in_file, header = 0)
        df['cur_pos'] = np.zeros(len(df)) - 1

        for i in len(df):
            if df.iloc[i,0] == 1:
                if i == 0: 
                    df.iloc[i,1] = pos 
                else: 
                    if (df.iloc[i-1,1] > 0):
                        df.iloc[i,1] = df.iloc[i,1] + pos 
                    elif (df.iloc[i-1,1] < 0): 
                        df.iloc[i,1] = 0 
                    else: 
                        df.iloc[i,1] = pos 

            elif df.iloc[i,0] == 0: 
                if i == 0: 
                    df.iloc[i,1] == 0
                else: 
                    df.iloc[i,1] == df.iloc[i-1,1]

            elif df.iloc[i,0] == -1: 
                if i == 0: 
                    df.iloc[i,1] = -pos 
                else: 
                    if (df.iloc[i-1,1] > 0):
                        df.iloc[i,1] = 0
                    elif (df.iloc[i-1,1] < 0): 
                        df.iloc[i,1] = df.iloc[i,1] - pos 
                    else: 
                        df.iloc[i,1] = -pos 
            
            df.to_csv('./Prediction/'+ out_file)
        return 
   


    
    def five_label_pos(in_file, out_file, pos_1, pos_2):
        '''
        5-label: 
        2: more wegiht, 1: less weight 
        1. If 2: 
            1.1. 2->2, add pos_2
            1.2. 2->1, add pos_1
            1.3. 2->0, keep pos
            1.4. 2->(-) Close 
        Similar 
        '''
        df = pd.read_csv('./Prediction/'+ in_file, header = 0)
        df['cur_pos'] = np.zeros(len(df)) - 1

        for i in len(df):
            if df.iloc[i,0] == 2:
                if i == 0: 
                    df.iloc[i,1] = pos_2
                else: 
                    if (df.iloc[i-1,1] > 0):
                        df.iloc[i,1] = df.iloc[i,1] + pos_2
                    elif (df.iloc[i-1,1] < 0): 
                        df.iloc[i,1] = 0 
                    else: 
                        df.iloc[i,1] = pos_2

            elif df.iloc[i,0] == 1:
                if i == 0: 
                    df.iloc[i,1] = pos_1
                else: 
                    if (df.iloc[i-1,1] > 0):
                        df.iloc[i,1] = df.iloc[i,1] + pos_1
                    elif (df.iloc[i-1,1] < 0): 
                        df.iloc[i,1] = 0 
                    else: 
                        df.iloc[i,1] = pos_1

            elif df.iloc[i,0] == 0: 
                if i == 0: 
                    df.iloc[i,1] == 0
                else: 
                    df.iloc[i,1] == df.iloc[i-1,1]

            elif df.iloc[i,0] == -1: 
                if i == 0: 
                    df.iloc[i,1] = -pos_1
                else: 
                    if (df.iloc[i-1,1] > 0):
                        df.iloc[i,1] = 0
                    elif (df.iloc[i-1,1] < 0): 
                        df.iloc[i,1] = df.iloc[i,1] - pos_1
                    else: 
                        df.iloc[i,1] = -pos_1
            
            elif df.iloc[i,0] == -2: 
                if i == 0: 
                    df.iloc[i,1] = -pos_2
                else: 
                    if (df.iloc[i-1,1] > 0):
                        df.iloc[i,1] = 0
                    elif (df.iloc[i-1,1] < 0): 
                        df.iloc[i,1] = df.iloc[i,1] - pos_2
                    else: 
                        df.iloc[i,1] = -pos_2
            
            df.to_csv('./Prediction/'+ out_file)
        return 
   
    
    
   

    