import numpy as np
import pandas as pd

df = pd.read_csv('F:\\Valerio\\Scripts and Plots\\ValerioTrials\\Range\\vehicle_data.csv')

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
