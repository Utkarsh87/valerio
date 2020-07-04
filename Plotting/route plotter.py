#!/usr/bin/env python
# coding: utf-8

"""
SCRIPT TO GENERATE THE OPTIMAL ROUTE BETWEEN USER FED SOURCE AND
DESTINATION POINTS CONSIDERING THE ELECTRIC VEHICLE'S CHARGING NEEDS.
"""

"""
Author: Utkarsh Kumar Singh
With help from: Prajwal Ranjan(Co-author), Sarthak Mahapatra(Charging station database).
"""

# Credits:
# Base code: https://ipython-books.github.io/147-creating-a-route-planner-for-a-road-network/
# Data(.shp file): https://mapcruzin.com/free-delhi-country-city-place-gis-shapefiles.htm



# ****************** IMPORTS ******************

# System libraries
import io
import os
import zipfile
import sys
import platform
if platform.system() == 'Windows':
    split_by = "\\"
else:
    split_by = "/"
# Note: Modern windows filesystems generally allow both / and \ in their paths and hence they can be used
# interchangeably, but for the few systems that don't; the above check is necessary.

cur = os.path.abspath(__file__) # get the path to this script
parent = cur.rsplit(split_by, 1)[0] # get parent dir of script
parent = parent.rsplit(split_by, 1)[0]

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, parent+split_by+"Geocoding")
sys.path.insert(2, parent+split_by+"Range")

# External libraries
    # For data wrangling
import numpy as np
import pandas as pd
import json
    # For working with the shapefile
import gdal
    # For working with graphs
import networkx as nx
    # Graphing and plotting utilities
import smopy
import matplotlib.pyplot as plt

# Author-defined libraries
from GeocodeAddress import geocoding
from RangeCalculation import getRange



# ****************** READ THE ROAD-NETWORK FILE(SHAPEFILE) ******************

# Unzip util
def unzip_file(name):
    """
    Arguement: filename
    """
    zipped_file = name
    zip_ref = zipfile.ZipFile(zipped_file, 'r')
    zip_ref.extractall('.')
    zip_ref.close()


# Unzip the road network file
unzip_file(parent+split_by+"Plotting"+split_by+"delhi_highway.zip")


# Read the shapefile into networkx object
g = nx.read_shp(parent+split_by+"Plotting"+split_by+"delhi_highway.shp")


# Graph might not be connected, this function will yield the largest connected subgraph.
def connected_component_subgraphs(G):
    """
    Arguement: graph G
    Return value: largest connected subgraph
    """
    for c in nx.connected_components(G):
        yield G.subgraph(c)

sgs = list(connected_component_subgraphs(g.to_undirected())) # sgs is a list of subgraphs


i = np.argmax([len(sg) for sg in sgs]) # i holds the location of the largest connected subgraph
sg = sgs[i] # sg is the subgraph of interest
# print(len(sg)) # number of nodes in this subgraph




# ****************** USER INPUTS: SOURCE AND DESTINATION POINTS ******************

# Input the source and destination points
print("Enter source: ")
address = input()
pos0 = geocoding(address)
print("Enter destination: ")
address = input()
pos1 = geocoding(address)
print("Source entered: ", pos0)
print("Destination entered: ", pos1)



# ****************** GRAPH WEIGHT METRIC: DISTANCE B/W THE NODES(IN KM) ******************

def get_path(n0, n1): 
    """
    Arguements: n0 and n1 are tuples(containing the latitude and longitude) of the 2 locations
    Return value: array of points linking the 2 given locations.
    """
    return np.array(json.loads(sg[n0][n1]['Json'])
                    ['coordinates'])


radius = 6372.8
def calcDist(lat0, lon0, lat1, lon1):
    # calculates and returns the Great-Circle Distance using the cosine formula(in km)

    """
    Arguements: takes 4 numbers, the latitudes and the longitudes of the points 
                between which distance is to be calculated.
    Return value: distance between the 2 points in km
    """
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    dlon = lon0 - lon1
    
    y = np.sqrt((np.cos(lat1) * np.sin(dlon)) ** 2 +
        (np.cos(lat0) * np.sin(lat1) - np.sin(lat0) *
         np.cos(lat1) * np.cos(dlon)) ** 2)
    x = np.sin(lat0) * np.sin(lat1) + \
        np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
    c = np.arctan2(y, x)
    
    return radius*c


# leverages the calcDist function to calculate the length between given 2 points
def get_path_length(path):
    return np.sum(calcDist(path[1:, 1], path[1:, 0], path[:-1, 1], path[:-1, 0])) # pass the lat and lon of the 2 points





# ****************** CHARGING STATION DATABASE(CSV FILE) ******************

# Get charging stations from the csv file
df = pd.read_csv(parent+split_by+"Plotting"+split_by+"ModifiedStations.csv")
df['location'] = list(zip(df.LATITUDE, df.LONGITUDE))

stations = []
for station in df.location:
    stations.append(station)
    
stations = np.array(stations) # Convert list to numpy array
stations = stations[:, ::-1]  # Flip the tuple from (lat, lon) to (lon, lat)

station_set = set()
for val in stations:
    val = val.tolist()
    val = tuple(val)
    station_set.add(val)
    
# print(station_set)  # Set of locations of the charging stations stored as tuples.(lon, lat)





# ****************** REDUCE SEARCH SPACE ******************

"""
Done by checking where the destination lies relative to the source and only looking in that
particular quadrant for charging stations.
"""

"""
Each of the four functions below will take in the source(src) and the destination(dest)
points as arguements and will return the possible stations that lie in the quadrant of
the destination point keeping the source at origin.
"""

# Check if dest lies in the first quadrant if the src is kept at the origin
def first_quadrant(src, dest):
    # have to find stations such that their lon >= lon of src && <= lon of dest and their lat >= lat of src and <= lat of dest
    lat_bound, lon_bound = pos1[0], pos1[1]
    possible_stations = []
    for val in station_set:
        if(val[1] <= lat_bound and val[0] <= lon_bound and val[1] >= src[0] and val[0] >= src[1]):
            possible_stations.append(val)
            
    return possible_stations

# Check if dest lies in the second quadrant if the src is kept at the origin
def second_quadrant(src, dest):
    lat_bound, lon_bound = pos1[0], pos1[1]
    possible_stations = []
    for val in station_set:
        if(val[1] <= lat_bound and val[0] >= lon_bound and val[1] >= src[0] and val[0] <= src[1]):
            possible_stations.append(val)
            
    return possible_stations

# Check if dest lies in the third quadrant if the src is kept at the origin
def third_quadrant(src, dest):
    lat_bound, lon_bound = pos1[0], pos1[1]
    possible_stations = []
    for val in station_set:
        if(val[1] >= lat_bound and val[0] >= lon_bound and val[1] <= src[0] and val[0] <= src[1]):
            possible_stations.append(val)
            
    return possible_stations

# Check if dest lies in the fourth quadrant if the src is kept at the origin
def fourth_quadrant(src, dest):
    lat_bound, lon_bound = pos1[0], pos1[1]
    possible_stations = []
    for val in station_set:
        if(val[1] >= lat_bound and val[0] <= lon_bound and val[1] <= src[0] and val[0] >= src[1]):
            possible_stations.append(val)
            
    return possible_stations

# pos0 src, pos1 dest
possible_stations = []
if(pos1[1] > pos0[1] and pos1[0] > pos0[0]):
    # first quadrant
    possible_stations = first_quadrant(pos0, pos1)
    # print("first")
    
elif(pos1[1] < pos0[1] and pos1[0] > pos0[0]):
    # second quadrant
    possible_stations = second_quadrant(pos0, pos1)
    # print("second")
    
elif(pos1[1] < pos0[1] and pos1[0] < pos0[0]):
    # third quadrant
    possible_stations = third_quadrant(pos0, pos1)
    # print("third")
    
elif(pos1[1] > pos0[1] and pos1[0] < pos0[0]):
    # fourth quadrant
    possible_stations = fourth_quadrant(pos0, pos1)
    # print("fourth")
    
# Debug util
# print("Number of stations inside the bounding box: ", len(possible_stations))




# ****************** UPDATE THE GRAPH OF THE ROAD NETWORK ******************

"""
Update the graph edges by assigning the weights of the edges, 
the actual distance between the 2 nodes in km.
"""

for n0, n1 in sg.edges:
    path = get_path(n0, n1) # will return numpy array of points, path is a list of lats and lons each returned in a list format, hence a list of lists.
    dist = get_path_length(path)
    sg.edges[n0, n1]['distance'] = dist # update step

# print(sg.edges)   # list of nested tuples.
# print(sg.nodes)   # list of nodes.
# print(len(sg))    # Number of nodes in this subgraph.

nodes = np.array(sg.nodes)

"""
Our requested positions might not be on the graph, find the locations on the graph closest to them
get all the nodes of the constructed graph into an array and find the point closest to target point.
"""

# Util function for getting the closest point that is a node in the graph.
def getClosestPointIndex(loc):
    """
    Arguements: loc: coordinates of a point
    Return value: returns the coordinate closest to the one given that is a node in the shapefile
    """
    loc_i = np.argmin(np.sum((nodes[:, ::-1] - loc)**2, axis=1))
    return loc_i

# Get the closest nodes in the graph. source_i and destination_i return index of the points closest to requested points in the nodes array
source_i = getClosestPointIndex(pos0)
destination_i = getClosestPointIndex(pos1)

# print("Source: ", nodes[source_i])
# print("Destination: ", nodes[destination_i])


# Debug util
if(len(possible_stations) > 0):
    station_dict = {}
    for station in possible_stations:
        station_i = np.argmin(np.sum((nodes - station)**2, axis=1))
        station_dict[station] = station_i
    # print("Station locations along with their indices :\n", station_dict)




# ****************** USER INPUT: BATTERY LEVEL AND VEHICLE MODEL ******************

rem_charge = float(input("Enter the remaining battery level: "))
can_travel, deets = getRange(rem_charge)

print(f'Vehicle: {deets[0]}, Range on full-charge: {deets[1]}')
range_on_full_charge = deets[1]
print(f"The car can travel {can_travel} km based on the current charge of {rem_charge}%")


"""
get the shortest path length from source to all the possible stations and then refine the 
list of possible stations down to only those that can be reached based on the current charge.
"""

refined_possible_stations = []
if(len(possible_stations) > 0):
    for val, index in station_dict.items():
        dist_to_station = nx.astar_path_length(sg, 
                                source = tuple(nodes[source_i]), 
                                target = tuple(nodes[index]),
                                weight = 'distance')
        if(dist_to_station < can_travel):
            refined_possible_stations.append(val)
            
    # if(len(refined_possible_stations) > 0):
        # print("Possible stations that can be visited based on current charge(in (longitude, latitude) form): ", refined_possible_stations)
        # print("Number of possible stations that can be visited based on current charge: ", len(refined_possible_stations))


# Distance from source to destination
def getDistanceToDestination():
    """
    Arguement: null
    Return value: Distance(in km) between the source and destination point
                  (as fetched from the closest nodes in the graph)
    """
    dist = nx.astar_path_length(sg, 
                                source = tuple(nodes[source_i]), 
                                target = tuple(nodes[destination_i]),
                                weight = 'distance')
    print(f'The distance from the source to destination is {round(dist, 3)}km')
    return dist


# Util function for finding the charging station closest to source point
def helper_StationClosestToSource():
    sourceDistToStation = 1e10
    closestStationToSource = []
    for val in stations:
        dist = calcDist(val[1], val[0], pos0[0], pos0[1])      # User entered values are (lat, lon) all others are (lon, lat)
        if dist < sourceDistToStation:
            closestStationToSource = val
            sourceDistToStation = dist
    print("Charging Station closest to source point entered: ", closestStationToSource)
    print("Distance of the closest charging station from source(in km): ", round(sourceDistToStation, 3))
    station_i = np.argmin(np.sum((nodes - closestStationToSource)**2, axis=1))
    
    length_path1 = nx.astar_path_length(sg, 
                source = tuple(nodes[source_i]), 
                target = tuple(nodes[station_i]),
                weight = 'distance')
    length_path2 = nx.astar_path_length(sg, 
                source = tuple(nodes[station_i]), 
                target = tuple(nodes[destination_i]),
                weight = 'distance')
    
    print(f'The vehicle will have to cover a distance of {round(length_path1+length_path2, 3)}km in total.')
    # print("Station closest to source point: ", nodes[station_i])
    
    return station_i

# Find the station closest to the source point
def getStationClosestToSource():
    dist = getDistanceToDestination()
    if(can_travel > dist):
        refuel = input("Your vehicle has enough charge to make it to the destination. Stop for a recharge anyway ?(YES/NO)")
        if(str.lower(refuel) == "yes"):
            station_i = helper_StationClosestToSource()
            
        elif(str.lower(refuel) == "no"):
            # route the user straight to the destination
            station_i = -1
            
    else:
        refuel = "yes"
        print("The vehicle will need to make a stop for recharging on the way to charging station. Finding the optimal route...")
        station_i = helper_StationClosestToSource()
        
    return station_i, refuel




# ****************** FIND THE OPTIMAL ROUTE ******************

if(len(refined_possible_stations) == 0):
    # no station close enough hence have to greedily look for the closest station for a recharge
    # i.e. no station inside the bounding box
    station_i, refuel = getStationClosestToSource()
    
elif(len(refined_possible_stations) > 0):
    # refined_possible_stations is the list of all stations the user can get to on remaining charge
    
    # maybe there is no need for a recharge, this will be in the case the dist from source to destination is lesser
    # than the remaining range, if so ask the user for a recharge anyway.

    dist = getDistanceToDestination()
    if(can_travel > dist):
        # they can get to the destination without a recharge, ask anyways
        refuel = input("Your vehicle has enough charge to make it to the destination. Stop for a recharge anyway ?(YES/NO)")
        if(str.lower(refuel) == "yes"):
            min_dist = 1e10
            for st in refined_possible_stations:
                st_i = station_dict[st] 
                dist1 = nx.astar_path_length(sg, 
                                        source = tuple(nodes[source_i]), 
                                        target = tuple(nodes[st_i]),
                                        weight = 'distance')
                dist2 = nx.astar_path_length(sg,
                                        source = tuple(nodes[st_i]),
                                        target = tuple(nodes[destination_i]),
                                        weight = 'distance')
                if(dist1+dist2 < min_dist):
                    station_i = st_i
                    min_dist = dist1+dist2
                    station_dist = dist1
                    destination_dist = dist2
#                 print("Distance: ", min_dist)
            
            # having calculated the choice of the station which minimises the total distance that will be covered
            # prompt the user again to reconsider their choice of going for a recharge if the "detour" is above
            # a certain threshold.
            
            length_path1 = nx.astar_path_length(sg, 
                            source = tuple(nodes[source_i]), 
                            target = tuple(nodes[station_i]),
                            weight = 'distance')
            length_path2 = nx.astar_path_length(sg, 
                            source = tuple(nodes[station_i]), 
                            target = tuple(nodes[destination_i]),
                            weight = 'distance')
            
            threshold = 1 # in km
            if(length_path1+length_path2-dist > threshold and can_travel > length_path1): # make the user re-consider the decision to go for recharging based on detour distance.
                print("The total distance that will have to be covered if you choose to go for a recharge is: ", round(length_path1+length_path2, 3))
                print(f'Hence the vehicle will have to cover an extra {round((length_path1+length_path2-dist), 3)}km due to this detour')
                refuel = input("Do you still want to recharge your vehicle on the way to the destination ?(YES/NO)")
            
                if(str.lower(refuel) == "no"):
                    station_i = -1
                    
            if(str.lower(refuel) == "yes"):
                print(f'We will have to cover the least possible distance if we go through the charging station close to {nodes[station_i]} which is {round(station_dist, 2)}km away from the source and {round(destination_dist, 2)}km away from the destination.')
        
        
        elif(str.lower(refuel) == "no"):
            station_i = -1
            
    else:
        print("The vehicle will need to make a stop for recharging on the way to the destination. Finding the optimal route...")
        # a recharge is needed, so out of the possible stations get the one which has lowest total path length to the destination.
        min_dist = 1e10
        for st in refined_possible_stations:
            st_i = station_dict[st] 
            dist1 = nx.astar_path_length(sg, 
                                    source = tuple(nodes[source_i]), 
                                    target = tuple(nodes[st_i]),
                                    weight = 'distance')
            dist2 = nx.astar_path_length(sg,
                                    source = tuple(nodes[st_i]),
                                    target = tuple(nodes[destination_i]),
                                    weight = 'distance')
            if(dist1+dist2 < min_dist):
                station_i = st_i
                min_dist = dist1+dist2
                station_dist = dist1
                destination_dist = dist2
        
        # check if the nearest charging station is out-of-range:
        if(station_dist > can_travel):
            print('''The vehicle will not be able to make it to the nearest charging station on current charge levels. You might consider installing a backup battery pack or charging the current battery enough to get to the nearest charging station.''')
            
            # calculate charge needed to get to the nearest charging station.
            charge_needed = (station_dist/range_on_full_charge)*100
            
            print(f'''If you wish to charge the current battery, charge it till {round(charge_needed, 2)}% to get to the nearest charging station at {nodes[station_i]}.''')
            
            print("After the battery is charged to the minimal level, enter the battery details again.")
            
        else:
            refuel = "yes"
            print(f'We will have to cover the least possible distance if we go through the charging station close to {nodes[station_i]} which is {round(station_dist, 2)}km away from the source and {round(destination_dist, 2)}km away from the destination.')



# Utility function to build a continuous path from an array of nodes.
def get_full_path(path):
    """Return the positions along a path. Appends the points in the correct order to be able to build a path."""
    p_list = []
    curp = None
    for i in range(len(path) - 1):
        p = get_path(path[i], path[i + 1])
        if curp is None:
            curp = p
        if (np.sum((p[0] - curp) ** 2) >
                np.sum((p[-1] - curp) ** 2)):
            p = p[::-1, :]
        p_list.append(p)
        curp = p[-1]
    return np.vstack(p_list)




# ****************** ROUTE PLOTTING ******************

# There are 2 cases to consider :
# 1. The vehicle goes for refuelling(by the user's choice(if the car has enough charge) or necessarily(lack of enough charge to take it to the destination directly))
# 2. The user chooses not to refuel and go to the destination directly instead.

# Instantiate the map(this step is common for both the cases)
m = smopy.Map(pos0, pos1, z=15, margin=0.95)


# Case 1: Refuelling

# Having chosen the charging station to be visited, build the route accordingly
if(str.lower(refuel) == "yes"):

    # Compute the shortest path from source to the charging station closest to source point.
    path1 = nx.astar_path(
        sg,
        source=tuple(nodes[source_i]),
        target=tuple(nodes[station_i]),
        weight='distance')  # Computes the shortest path (weighted by distance).
    
    length_path1 = nx.astar_path_length(sg, 
                            source = tuple(nodes[source_i]), 
                            target = tuple(nodes[station_i]),
                            weight = 'distance')

    # Compute the shortest path from charging station closest to source point to destination.
    path2 = nx.astar_path(
        sg,
        source=tuple(nodes[station_i]),
        target=tuple(nodes[destination_i]),
        weight='distance')
    
    length_path2 = nx.astar_path_length(sg, 
                            source = tuple(nodes[station_i]), 
                            target = tuple(nodes[destination_i]),
                            weight = 'distance')

    # path = path1+path2 # is the final path from source->nearest charging station->destination
    # print("Path 1: ", path1)
    # print("Path 2: ", path2)
    # print("Path: ", path)
    # The path variable now contains the list of edges that form the shortest path between requested positions
    # print(len(path))


if(str.lower(refuel) == "yes"):
    # Get the co-ordinates of the approximate source, charging station and destination:
    print("Source: ", nodes[source_i])
    print("Station: ", nodes[station_i])
    print("Destination: ", nodes[destination_i])


# Edge case: charging station right where the source or destination point is.
if(str.lower(refuel) == "yes"):
    if source_i != station_i:
        linepath = get_full_path(path1)
        a, b = m.to_pixels(linepath[:, 1], linepath[:, 0])

    else:
        print("There's a charging station right where you are !")
        
    if station_i != destination_i:
        linepath = get_full_path(path2)
        x, y = m.to_pixels(linepath[:, 1], linepath[:, 0])
        
    else:
        print("There's a charging station right next to your destination point!")

if(str.lower(refuel) == "yes"):
    
    ax = m.show_mpl(figsize=(8, 8))

    # Mark our positions.(Source->Station->Destination : RBG)
    # Source will be marked with '*' in red.
    if source_i == station_i:
        ax.plot(x[0], y[0], 'or', ms=10, marker='*')
        ax.plot(x[-1], y[-1], 'og', ms=10)
        
    elif destination_i == station_i:
        ax.plot(a[0], b[0], 'or', ms=10, marker='*')
        ax.plot(a[-1], b[-1], 'og', ms=10)    
        
    else:
        ax.plot(a[0], b[0], 'or', ms=10, marker='*')
        ax.plot(x[0], y[0], 'ob', ms=10, marker='^')
        ax.plot(x[-1], y[-1], 'og', ms=10)       
    
    

    # Mark the 2 paths, first from source to closest charging station and then from there on to the destination.
    if source_i != station_i:
        ax.plot(a, b, '-k', lw=2) # Source to station is marked with black.
    if station_i != destination_i:
        ax.plot(x, y, '-y', lw=2) # Station to destination is marked with yellow.

    plt.show()
    # NOTE: If the black and yellow lines overlap, yellow will be seen.



# Case 2: The user chooses not to refuel their vehicle and go to the destination directly instead).
if(str.lower(refuel) == "no"):
    # Compute the shortest path from source to the destination.
    path = nx.astar_path(
        sg,
        source=tuple(nodes[source_i]),
        target=tuple(nodes[destination_i]),
        weight='distance')  # Computes the shortest path (weighted by distance).

if(str.lower(refuel) == "no"):
    print("Source: ", nodes[source_i])
    print("Destination: ", nodes[destination_i])

if(str.lower(refuel) == "no"):
    linepath = get_full_path(path)
    x, y = m.to_pixels(linepath[:, 1], linepath[:, 0])

if(str.lower(refuel) == "no"):
    ax = m.show_mpl(figsize=(8, 8))
    ax.plot(x[0], y[0], 'or', ms=10, marker='*')
    ax.plot(x[-1], y[-1], 'og', ms=10)
    ax.plot(x, y, '-k', lw=2)
    plt.show()