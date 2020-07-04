import numpy as np
import pandas as pd


import platform
if platform.system() == 'Windows':
    split_by = "\\"
else:
    split_by = "/"
# Note: Modern windows filesystems generally allow both / and \ in their paths and hence they can be used
# interchangeably, but for the few systems that don't; the above check is necessary.

import os
cur = os.path.abspath(__file__)
cur = cur.rsplit(split_by, 1)[0]


df = pd.read_csv(cur+split_by+"vehicle_data.csv")
car_list = list(df['VEHICLE NAME'])
range_list = list(df['DRIVING RANGE (IN KM)'])

def RemainingRange(battery_level):
    print('Please select the Model: \n')
    for car in car_list:
        print(str(car_list.index(car)+1) + "\t"+car+"\n")
    choice = int(input('Enter Here: '))
    car = car_list[choice-1]
    rang = range_list[choice-1]
    remaining_range = (rang*battery_level)/100
    
    return remaining_range, (car, rang)

def getRange(rem_battery):
    rem, deets = RemainingRange(rem_battery)
    return rem, deets
