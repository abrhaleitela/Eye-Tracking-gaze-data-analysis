# -*- coding: utf-8 -*-
#! /usr/bin/env python3

""" main.py: Eye Tracking Project """

__author__   = "Eye tracking 2019, 4. period, Group 10"
__version__  = "1.0.0"

import os
import re
import sys
import csv
import numpy as np
from matplotlib import pyplot
import math

# Constants used in the I-DT algorithm, 
# Note that if the input data is changed then this constants might need to be tuned

DESPERSION_MAX      = 80 
DURATION_MIN        = 0.1   # In seconds
SAMPLING_FREQUENCY  = 1000  # In Hertz
RADIUS              = 40 
HORIZONTAL = 97
VERTICAL = 56

def loadData():
    file = open('train.csv', 'r')
    data = list(csv.reader(file, delimiter=','))
    return (data)

def dataPreprocessing(data):
    data_set_dic = {}
    s10_ids = s20_ids = s30_ids = s6_ids = s16_ids = s26_ids = 0
    for row in data:
        if (str(row[0]) == "s10"):
            # Numbering order begining from 1 (Pascal)
            s10_ids = s10_ids + 1 
            dict_id = "s10_" + str(row[1]) + "_" + str(s10_ids)
        elif str(row[0]) == "s20": 
            s20_ids = s20_ids + 1
            dict_id = "s20_" + str(row[1]) + "_" + str(s20_ids)
        elif str(row[0]) == "s30":
            s30_ids = s30_ids + 1
            dict_id = "s30_" + str(row[1]) + "_" + str(s30_ids)
        elif str(row[0]) == "s6" :
            s6_ids = s6_ids + 1
            dict_id = "s6_" + str(row[1]) + "_" + str(s6_ids)
        elif str(row[0]) == "s16" :
            s16_ids = s16_ids + 1
            dict_id = "s16_" + str(row[1]) + "_" + str(s16_ids)
        elif str(row[0]) == "s26":
            s26_ids = s26_ids + 1
            dict_id = "s26_" + str(row[1]) + "_" + str(s26_ids)
        else:
            continue
         
        row_data = np.array(row[2:], dtype=float)
        row_data = row_data.reshape(int((len(row_data)/2)), 2)
        data_set_dic[dict_id] = row_data
    # pyplot.scatter(data_set_dic['s20_false_1'][:,[0]], data_set_dic['s20_false_1'][:,[1]])
    return(data_set_dic)

def despertion(dmin_x, dmax_x,dmin_y,dmax_y):
    d = (dmax_x - dmin_x) + (dmax_y - dmin_y)
    if(d <= DESPERSION_MAX):
        return True
    else:
        return False

def fixationDetection(data): 
    #Initializtion of variables
    window    = []
    centroids = []
    dmax_x = -1 * math.inf 
    dmax_y = -1 * math.inf
    dmin_x = math.inf
    dmin_y = math.inf
    for point in data:
        # Assign max and min values in the window
        if(point[0] < dmin_x):
            dmin_x = point[0]
        if(point[0] > dmax_x):
            dmax_x = point[0]
        if(point[1] < dmin_y):
            dmin_y = point[1]
        if(point[1] > dmax_y):
            dmax_y = point[1]
        
        # Check Despertion formula to add a point in the window
        if(despertion(dmin_x, dmax_x, dmin_y, dmax_y)):
            window.append([point[0],point[1]])
        else:
            n = len(window)
            time = n/SAMPLING_FREQUENCY
            # When the condition below is True it means we have detected one fixtation event
            if(time >= DURATION_MIN):
                # Find the centroid of the window(current fixation event) and add it to the centroids array. 
                window_np = np.array(window)
                sum_xy = np.sum(window_np, axis = 0) 
                x_avg = sum_xy[0] / n
                y_avg = sum_xy[1] / n
                x_deg = x_avg / HORIZONTAL
                y_deg = y_avg / VERTICAL
                centroid = [x_deg, y_deg, time]
                centroids.append(centroid)
            #else(It is not fixtation):
                # Other events
            #Reset all variables when we destroy our window
            window = []
            dmax_x = -1 * math.inf
            dmax_y = -1 * math.inf
            dmin_x = math.inf
            dmin_y = math.inf   
      
# =============================================================================
#     #Plot the result
#     pyplot.scatter(data[:,[0]], data[:,[1]])
#     for point in centroids:
#         pyplot.scatter(point[0],point[1],color='black')
#         center = pyplot.Circle((point[0],point[1]), RADIUS, fill = False, color ='red' )
#         pyplot.gcf().gca().add_artist(center)
#     pyplot.title("Output Data[Fixtation events detected]")
#     pyplot.xlabel("X axis")
#     pyplot.ylabel("Y axis")
#     pyplot.show()
# =============================================================================
    
    return(centroids)

def detectCentroids(data):
    returnVal = {}
    returnVal['s26_false'] = []
    returnVal['s10_false'] = []
    returnVal['s16_false'] = []
    returnVal['s6_false'] = []
    returnVal['s30_false'] = []
    returnVal['s20_false'] = []
    returnVal['s26_true'] = []
    returnVal['s10_true'] = []
    returnVal['s16_true'] = []
    returnVal['s6_true'] = []
    returnVal['s30_true'] = []
    returnVal['s20_true'] = []
    for dict_id in data:
        centroids = fixationDetection(data[dict_id]) 
        returnVal[dict_id[0:dict_id.rfind('_')]].append(centroids)
    return(returnVal)

# ============ TASK 2 functions ===============

def calculateMeanFixationDuration(data):

    totalSum       = float(0)
    totalDurations = int(0)

    for trial in data:
        for fixation in trial:
            totalSum       += fixation[2]
            totalDurations += 1

    return (totalSum/totalDurations)

def calculateStandardDeviationMeanFixationDuration(data, meanValue):

    totalSum     = float(0)
    totalRecords = int(0)

    for trial in data:
        for fixation in trial:
            totalSum     += (abs(fixation[2] - meanValue)**2)
            totalRecords += 1

    return math.sqrt(totalSum/totalRecords)

def calculateMeanSaccadeAmplitudes(data):

    totalSum        = float(0)
    totalAmplitudes = int(0)

    for trial in data:
        for index in range(len(trial) - 1):
            totalSum        += calculateDistanceTwoPoints(trial[index][2], trial[index + 1][2], trial[index][1], trial[index + 1][1])
            totalAmplitudes += 1

    return (totalSum/totalAmplitudes)

def calculateStandardDeviationMeanSaccadeAmplitudes(data, meanValue):

    totalSum     = float(0)
    totalRecords = int(0)

    for trial in data:
        for index in range(len(trial) - 1):
            totalSum     += (abs(calculateDistanceTwoPoints(trial[index][2], trial[index + 1][2], trial[index][1], trial[index + 1][1]) - meanValue)**2)
            totalRecords += 1

    return math.sqrt(totalSum/totalRecords)

def calculateDistanceTwoPoints(x1, x2, y1, y2):

    return(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

def calculateStatisticsOneSubject(subjectName, dataOfSubjectTrue, dataofSubjectFalse):
    """ 
    Returns list of: MFD_true MFD_SD_true MFD_false MFD_SD_false MSA_true 
    MSA_SD_true MSA_false MSA_SD_false MFD_overall MFD_overall_SD MSA_overall 
    MSA_overall_SD.
    """

    subjectStatistics = list()

    # Append the subject name
    subjectStatistics.append(subjectName)

    # Calculating MFD for True and False
    subjectStatistics.append(calculateMeanFixationDuration(dataOfSubjectTrue))
    subjectStatistics.append(calculateStandardDeviationMeanFixationDuration(dataOfSubjectTrue, subjectStatistics[1]))
    subjectStatistics.append(calculateMeanFixationDuration(dataofSubjectFalse))
    subjectStatistics.append(calculateStandardDeviationMeanFixationDuration(dataofSubjectFalse, subjectStatistics[3]))

    # Calculating MSA for True and False
    subjectStatistics.append(calculateMeanSaccadeAmplitudes(dataOfSubjectTrue))
    subjectStatistics.append(calculateStandardDeviationMeanSaccadeAmplitudes(dataOfSubjectTrue, subjectStatistics[5]))
    subjectStatistics.append(calculateMeanSaccadeAmplitudes(dataofSubjectFalse))
    subjectStatistics.append(calculateStandardDeviationMeanSaccadeAmplitudes(dataofSubjectFalse, subjectStatistics[7]))

    # Calculating MSA and MFD for Overall
    subjectStatistics.append(calculateMeanFixationDuration(dataOfSubjectTrue + dataofSubjectFalse))
    subjectStatistics.append(calculateStandardDeviationMeanFixationDuration(dataOfSubjectTrue + dataofSubjectFalse, subjectStatistics[9]))
    subjectStatistics.append(calculateMeanSaccadeAmplitudes(dataofSubjectFalse + dataOfSubjectTrue))
    subjectStatistics.append(calculateStandardDeviationMeanSaccadeAmplitudes(dataofSubjectFalse + dataOfSubjectTrue, subjectStatistics[11]))

    return subjectStatistics

def printOutputToCsv(fileName, data):
    outputFile    = open(fileName, "w")
    csvOutputFile = csv.writer(outputFile, delimiter=',')
    for subject in data:
        csvOutputFile.writerow(subject)

    outputFile.close()

def main(argc, argv):
    """ Main function of the script. """

    # ====== TASK 1 ======
    inputData = loadData()
    dataSet   = dataPreprocessing(inputData)
    inputData = None

    """ centroids structure::: ==>  It is a dictinary as the following structure
    Structure: 
    dict_id: array([xcenter1,ycenter1,timeduration])  
    
    Example:  
    's20_false_27' : [[789.9555555555553, 805.3283597883598, 0.189], .... , [-29.039503546099297, 326.2302836879435, 0.423]]  

    s20_false : [[[]]]
    s20_true  : 

    dict_id is meaningful... it represents sample_id [s10] concatenated with underscore[_] 
                            concatenated with known[true/false] concatenated with underscore[_]
                            concatenated with counter integer [this counter counts number of samples having same sample id but not necessarily true or false.
    
    
    This way you can extract information you might need from the dict_id it self.
    [by splitting the dict_id based on underscore you will get [sample id, know,counter for each sample]   

    """

    centroids = detectCentroids(dataSet)
    dataSet   = None
    # print(centroids)

    # ====== TASK 2 ======

    print(sorted(centroids.keys()))

    #print(centroids["s10_true"][0][1:5])
    #print(centroids["s10_false"][0][1:5])
    #print(centroids["s16_true"][0][1:5])
    #print(centroids["s16_false"][0][1:5])
    #print(centroids["s20_true"][0][1:5])
    #print(centroids["s20_false"][0][1:5])
    #print(centroids["s26_true"][0][1:5])
    #print(centroids["s26_false"][0][1:5])
    #print(centroids["s30_true"][0][1:5])
    #print(centroids["s30_false"][0][1:5])


    subjectList = list()

    subjectList.append(calculateStatisticsOneSubject("s6", centroids["s6_true"], centroids["s6_false"]))    
    subjectList.append(calculateStatisticsOneSubject("s10", centroids["s10_true"], centroids["s10_false"]))
    subjectList.append(calculateStatisticsOneSubject("s16", centroids["s16_true"], centroids["s16_false"]))
    subjectList.append(calculateStatisticsOneSubject("s20", centroids["s20_true"], centroids["s20_false"]))    
    subjectList.append(calculateStatisticsOneSubject("s26", centroids["s26_true"], centroids["s26_false"]))
    subjectList.append(calculateStatisticsOneSubject("s30", centroids["s30_true"], centroids["s30_false"]))

    print(subjectList)

    # ====== TASK 3 ======

    printOutputToCsv("group10.csv", subjectList)

    # ====== TASK 4 ======

    # GRAPHS -> SALSENG

if __name__ == "__main__":
	""" Calls the function main()."""
	main(len(sys.argv), sys.argv)

# End of file: main.py
