################################################################################
#
# Author: Aram Asdourian
#
# Date: 2/25/2026
#
# Description:
# Algorithmic Time‑Series Segmentation and Condition Analysis Using Water Pump RUL Dataset
################################################################################
import random
import sys

###############################Importing Dataset################################

# initializes the dataset with a dictionary for row and column
dataset = {'row' : {}, 'column' : {}}

#opens the csv file
with open('rul_hrs.csv','r') as file:
    #adds the headers as keys to the columns dictionary
    headers = file.readline().strip('\n').split(',')
    dataset['column'] = {header : [] for header in headers}

    #iterates over the first 10000 rows
    for i in range(0,10000):
        #adds the row to the row dictionary
        row = file.readline().strip('\n').split(',')
        dataset['row'][row[0]] = row[1:]

        #adds the data from the row under each column
        for header, value in zip(headers,row):
            dataset['column'][header].append(value)

#gets the 3 quantiles
Q10 = float(dataset['column']['rul'][int(len(dataset['row'])*.90)])
Q40 = float(dataset['column']['rul'][int(len(dataset['row'])*.60)])
Q90 = float(dataset['column']['rul'][int(len(dataset['row'])*.10)])

#creates a condition column
dataset['column']['condition'] = []

#assigns a condition to each row depending on the rul in relation to the quantiles
for row in dataset['row'].values():
    rul = float(row[51])

    #compares rul to quantiles
    if rul < Q10:
        row.append("Extremely Low RUL")
        dataset['column']['condition'].append("Extremely Low RUL")
    elif Q10 <= rul < Q40:
        row.append("Moderately Low RUL")
        dataset['column']['condition'].append("Moderately Low RUL")
    elif Q40 <= rul < Q90:
        row.append("Moderately High RUL")
        dataset['column']['condition'].append("Moderately High RUL")
    elif Q90 <= rul:
        row.append("Extremely High RUL")
        dataset['column']['condition'].append("Extremely High RUL")
################################################################################

#####################################Task-1#####################################
def segmentate(data):
    #calculates length and midpoint
    length = len(data)
    midpoint = length//2

    #variance must be found
    #first finds mean
    mean = 0
    for num in data:
        mean += float(num)
    mean = mean/length

    #then calculates squared deviance
    deviations = []
    for num in data:
        deviations.append((float(num)-mean)**2)

    #finally finds variance
    var = sum(deviations)/length

    #compares variance with threshold and either divides the segment or marks as stable
    if var > 1:
        #unstable, break into segments
        leftSegment = segmentate(data[:midpoint])
        rightSegment = segmentate(data[midpoint:])
    else:
        #stable, return 1
        return 1

    #return the total number of segments
    return leftSegment + rightSegment

#returns a dictionary of the complexity of each time series
segmentComplexity = {key : None for key in dataset['column']['']}
for timeseries in segmentComplexity:
    # randomly sample for 10 sensors and call segmentate
    timeseriesData = random.sample(dataset['row'][timeseries][1:51], 10)
    segmentComplexity[timeseries] = segmentate(timeseriesData)


print("Timeseries, segments, RUL class: ")
for i in list((dataset['row'].keys()))[:3]:
    print(i,segmentComplexity[i],dataset['column']['condition'][int(i)])
print("...")
for i in list((dataset['row'].keys()))[-3:]:
    print(i,segmentComplexity[i],dataset['column']['condition'][int(i)])
################################################################################

#####################################Task-2#####################################
#clustering reference: https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/
def cluster(data, clusternum):
    #returns the data if it reaches the number of clusters
    if clusternum <= 1:
        return[data]
    #finds the centers of the 2 new clusters
    cluster1Center = min(data.values())
    cluster2Center = max(data.values())

    #initializes the new clusters
    cluster1 = {}
    cluster2 = {}

    #fills the new clusters based on which number is closer to the center
    for key in data:
        if abs(cluster1Center-data[key]) < abs(cluster2Center-data[key]):
            cluster1[key] = data[key]
        else:
            cluster2[key] = data[key]

    #recursively calls cluster on the newly created clusters, clusternum is halved
    leftClusters = cluster(cluster1, clusternum / 2)
    rightClusters = cluster(cluster2, clusternum / 2)

    #returns the new clusters
    return leftClusters + rightClusters

#splits the time segments into 4 clusters based off the number of segments
clusters = cluster(segmentComplexity, 4)
clusterClass = {key : "" for key in range(0,len(clusters))}

#iterates over each cluster and finds its dominant class
for i in range(0,len(clusters)):
    cluster = clusters[i]
    ExLow = 0
    ModLow = 0
    ModHigh = 0
    ExHigh = 0

    #iterates over each timeseries in the cluster and counts which class appears most
    for timeseries in cluster:
        condition = dataset['column']['condition'][int(timeseries)]
        if condition == "Extremely Low RUL":
            ExLow += 1
        elif condition == "Moderately Low RUL":
            ModLow += 1
        elif condition == "Moderately High RUL":
            ModHigh += 1
        elif condition == "Extremely High RUL":
            ExHigh += 1

    #assigns a class to the cluster based off the dominant class
    if max([ExLow,ModLow,ModHigh,ExHigh]) == ExLow:
        clusterClass[i] = "Extremely Low RUL"
    elif max([ExLow,ModLow,ModHigh,ExHigh]) == ModLow:
        clusterClass[i] = "Moderately Low RUL"
    elif max([ExLow,ModLow,ModHigh,ExHigh]) == ModHigh:
        clusterClass[i] = "Moderately High RUL"
    elif max([ExLow,ModLow,ModHigh,ExHigh]) == ExHigh:
        clusterClass[i] = "Extremely High RUL"

#prints class of each cluster
print("\nCluster, Dominant Class:")
for cluster in clusterClass:
    print(cluster, clusterClass[cluster])

################################################################################

#####################################Task-3#####################################
#kadane reference: https://www.geeksforgeeks.org/dsa/largest-sum-contiguous-subarray/
def kadane(data):
    #calculates the absolute first difference
    difference = []
    for i in range(len(data)):
        difference.append(abs(float(data[i])-float(data[i-1])))

    #finds and subtracts the mean from each difference
    mean = sum(difference) / len(difference)
    variance = []
    for num in difference:
        variance.append(num-mean)

    #finds the maximum subset and tracks the start and endpoints
    maxNow = variance[0]
    maxEnd = variance[0]
    start = 0
    end = 0
    currentStart = 0
    for i in range(1,len(variance)):

        #checks if the subset is better to extend or leave alone
        if variance[i] > maxEnd + variance[i]:
            maxEnd = variance[i]
            currentStart = i
        else:
            maxEnd += variance[i]

        #checks which subset is better
        if maxEnd > maxNow:
            maxNow = maxEnd
            start = currentStart
            end = i

    #returns the start and endpoints of the max subset
    return start,end

#finds the dominnat class of each max subset
SubsetClass = {}
for sensor in headers[2:-1]:
    #gets the start and end points from the max subset
    start,end = kadane(dataset['column'][sensor])
    #creates a dictionary with the time series number and RUL category
    maxSubset = {key : value for key,value in zip(list(dataset['row'].keys())[start:end],list(dataset['column']['condition'][start:end]))}

    ExLow = 0
    ModLow = 0
    ModHigh = 0
    ExHigh = 0

    # iterates over each timeseries in the cluster and counts which class appears most
    for condition in maxSubset.values():
        if condition == "Extremely Low RUL":
            ExLow += 1
        elif condition == "Moderately Low RUL":
            ModLow += 1
        elif condition == "Moderately High RUL":
            ModHigh += 1
        elif condition == "Extremely High RUL":
            ExHigh += 1

    # assigns a class to the subset based off the dominant class
    if max([ExLow, ModLow, ModHigh, ExHigh]) == ExLow:
        SubsetClass[sensor] = [(start,end),"Extremely Low RUL"]
    elif max([ExLow, ModLow, ModHigh, ExHigh]) == ModLow:
        SubsetClass[sensor] = [(start,end),"Moderately Low RUL"]
    elif max([ExLow, ModLow, ModHigh, ExHigh]) == ModHigh:
        SubsetClass[sensor] = [(start,end),"Moderately High RUL"]
    elif max([ExLow, ModLow, ModHigh, ExHigh]) == ExHigh:
        SubsetClass[sensor] = [(start,end),"Extremely High RUL"]


print("\nSensor, max subset range, dominant class:")
for sensor in list(SubsetClass.items())[:3]:
    print(sensor[0], sensor[1][0], sensor[1][1])
print("...")
for sensor in list(SubsetClass.items())[-3:]:
    print(sensor[0], sensor[1][0], sensor[1][1])
################################################################################

print("\n\n")
userInput = -1
while userInput != '0':
    userInput = input("Enter 1 to print full segment data, 2 to print full subset data, or 0 to exit.")
    if userInput == '1':
        print("Timeseries, segments, RUL class: ")
        for i in list((dataset['row'].keys())):
            print(i, segmentComplexity[i], dataset['column']['condition'][int(i)])
    elif userInput == '2':
        print("\nSensor, max subset range, dominant class:")
        for sensor in list(SubsetClass.items()):
            print(sensor[0], sensor[1][0], sensor[1][1])
    elif userInput == '0':
        sys.exit()
    else:
        print("\nInvalid input, try again. ")
