from info_gain import info_gain 
import scipy.stats as sts
import numpy as np

def get_level(feature, level_num):
    level = [s[level_num -1] for s in feature ]
    return level
# process before add into package 
# Feature_1: get each price and volume 
f1 = np.loadtxt('f1.txt', dtype=int)
Pa1 = get_level(f1, 1)
Pa2 = get_level(f1, 2) 
Pa3 = get_level(f1, 3)
Pa4 = get_level(f1, 4)
Pa5 = get_level(f1, 5)
print(Pa1)
y = np.loadtxt('y_train.txt', dtype=int)
# Feature Set V2: Price Difference + Mid-price 


# Example of color to indicate whether something is fruit or vegatable
'''produce = ['apple', 'apple', 'apple', 'strawberry', 'eggplant']
fruit   = [ True  ,  True  ,  True  ,  True       ,  False    ]
colour  = ['green', 'green', 'red'  , 'red'       , 'purple'  ]
# ig  = info_gain.info_gain(label_y, feature)
ig  = info_gain.info_gain(fruit, colour)
iv  = info_gain.intrinsic_value(fruit, colour)
igr = info_gain.info_gain_ratio(fruit, colour)
print(ig, iv, igr)'''

# KL Divergency 
'''KL1 = sts.entropy([1/2, 1/2], qk=[9/10, 1/10])
KL2 = sts.entropy([9/10, 1/10], qk=[1/2, 1/2])
print(KL1)
print(KL2)'''
KL1 = sts.entropy(Pa1, y)
print(KL1)

# Differential entropy? 
# sts.differential_entropy