# -*- coding: utf-8 -*-
#! /usr/bin/env python3

""" main.py: Eye Tracking Final Project. """

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

DESPERSION_MAX      = 2     # In degrees
DURATION_MIN        = 0.1   # In seconds
SAMPLING_FREQUENCY  = 1000  # In Hertz
RADIUS              = 0.5 
HORIZONTAL          = 97
VERTICAL            = 56
DEGREES             = [HORIZONTAL, VERTICAL]

# ============================================================================
# = FUNCTIONS ================================================================
# ============================================================================

def loadData():
    """ Loading the input file "train.csv". """

    file = open('train.csv', 'r')
    data = list(csv.reader(file, delimiter=','))
    return (data)

def dataPreprocessing(data):
    """ Loades the data based on given subjects IDs. """

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

def convertUnitsToDegree(data):
    """ Converts units to degrees. """

    for ids in data:
        data[ids] = data[ids]/DEGREES
    return(data)

def despertion(dmin_x, dmax_x,dmin_y,dmax_y):
    """ Checks despertion maximum settings. """

    d = (dmax_x - dmin_x) + (dmax_y - dmin_y)
    if(d <= DESPERSION_MAX):
        return True
    else:
        return False

def fixationDetection(data): 
    """ Detects the fixation points. """

    # Initializtion of the variables
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
                centroid = [x_avg, y_avg, time]
                centroids.append(centroid)
            """else: (It is not fixtation):
                 Other events
            Reset all variables when we destroy our window """
            window = []
            dmax_x = -1 * math.inf
            dmax_y = -1 * math.inf
            dmin_x = math.inf
            dmin_y = math.inf         
# =============================================================================
#     # Plot the result
#     pyplot.plot(data[:,[0]], data[:,[1]])
#     for point in centroids:
#         pyplot.scatter(point[0],point[1],color='black')
#         center = pyplot.Circle((point[0],point[1]), RADIUS, fill = False, color ='red' )
#         pyplot.gcf().gca().add_artist(center)
#     pyplot.title("Output Data[Fixtation events detected]")
#     pyplot.xlabel("X axis in [°]")
#     pyplot.ylabel("Y axis in [°]")
#     pyplot.show()
# =============================================================================
    return(centroids)

def detectCentroids(data):
    """ Detects the centroids coordinates of the fixations. """

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
    """ Calculates the mean of fixation duration. """

    totalSum       = float(0)
    totalDurations = int(0)

    for trial in data:
        for fixation in trial:
            totalSum       += fixation[2]
            totalDurations += 1

    if(totalDurations == 0):
        return 0            

    return (totalSum/totalDurations)

def calculateStandardDeviationMeanFixationDuration(data, meanValue):
    """ Calculates the standard deviations of fixation duration. """

    totalSum     = float(0)
    totalRecords = int(0)

    for trial in data:
        for fixation in trial:
            totalSum     += (abs(fixation[2] - meanValue)**2)
            totalRecords += 1

    if(totalRecords == 0):
        return 0

    return math.sqrt(totalSum/totalRecords)

def calculateMeanSaccadeAmplitudes(data):
    """ Calculates the mean of saccade amplitudes. """

    totalSum        = float(0)
    totalAmplitudes = int(0)

    for trial in data:
        for index in range(len(trial) - 1):
            totalSum        += calculateDistanceTwoPoints(trial[index][1], trial[index + 1][1], trial[index][0], trial[index + 1][0])
            totalAmplitudes += 1

    if(totalAmplitudes == 0):
        return 0

    return (totalSum/totalAmplitudes)

def calculateStandardDeviationMeanSaccadeAmplitudes(data, meanValue):
    """ Calculates the standard deviation of the saccade amplitudes. """

    totalSum     = float(0)
    totalRecords = int(0)

    for trial in data:
        for index in range(len(trial) - 1):
            totalSum     += (abs(calculateDistanceTwoPoints(trial[index][1], trial[index + 1][1], trial[index][0], trial[index + 1][0]) - meanValue)**2)
            totalRecords += 1

    if(totalRecords == 0):
        return 0

    return math.sqrt(totalSum/totalRecords)

def calculateDurationAndCount(data):
    """ Unused. Just for statistical purposes. """
    """ Calculates the duration and count of fixations. """

    totalDuration      = int(0)
    totalDurationCount = int(0)

    for trial in data:
        totalDurationCount += len(trial)

        for fixation in trial:
            totalDuration += fixation[2]
            print(fixation[2])

    return [totalDuration/len(data), totalDurationCount/len(data)]

def calculateMeanTwoValues(value1, value2):
    """ Calculates simple aritmetic mean of two values. """

    return (value1 + value2) / 2

def calculateDistanceTwoPoints(x1, x2, y1, y2):
    """ Returns the euclidian distance between two points. """

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
    subjectStatistics.append(calculateMeanTwoValues(subjectStatistics[1], subjectStatistics[3]))
    subjectStatistics.append(calculateMeanTwoValues(subjectStatistics[2], subjectStatistics[4]))
    subjectStatistics.append(calculateMeanTwoValues(subjectStatistics[5], subjectStatistics[7]))
    subjectStatistics.append(calculateMeanTwoValues(subjectStatistics[6], subjectStatistics[8]))

    # Calculating the time spent on durations and their count (True, False)
    # subjectStatistics.append(calculateDurationAndCount(dataOfSubjectTrue))
    # subjectStatistics.append(calculateDurationAndCount(dataofSubjectFalse))

    return subjectStatistics

def drawGraph(subjectList,N):
    """ Draws the graphs. """

    # Graph which compares fixation durations against the users
    labels =[1,2]
    colors = ['red','red', 'blue','blue','teal', 'teal', 'yellow','yellow', 'green','green','gray', 'gray']
    MFD_std = []#Error
    MSA_std = []
    MFD_means = []
    MSA_means = []
    i = 3
    s = 0
    e = 2
    known = ['true','false']
    Agg_MSA_err = Agg_MFD_err = [0, 0] 
    for sub in subjectList:
        MFD_means.append(sub[1])
        MFD_means.append(sub[3])
        MFD_std.append(sub[2])
        MFD_std.append(sub[4])
        nTrue = N[sub[0]+'_'+known[0]]
        nFalse = N[sub[0]+'_'+known[1]]
        n = [math.sqrt(nTrue), math.sqrt(nFalse)]
        MFD_std_error = np.divide(MFD_std,n)
        Agg_MFD_err = Agg_MFD_err + MFD_std_error
        pyplot.bar(labels,MFD_means, color = colors[s:e], yerr = MFD_std_error, alpha = 0.5, align = 'center', capsize = 5)
        i,j = labels
        s=s+2
        e=e+2
        labels =[i + 2, j + 2]
        MFD_means = []
        MFD_std = []
        MFD_std_error  = []
        n = []
    pyplot.title("MFD bar charts for different samples(one sample has same color: First - True: Second - False)" )
    pyplot.legend(('S6','S10','S16','S20','S26','S30'), loc = 'upper right')
    pyplot.ylim(top = 0.5)
    pyplot.xlabel('Labels as in the legend')
    pyplot.ylabel('Mean of MFD points')
    pyplot.show()

    # Graph which compares saccade amplitudes against the users    
    labels =[1,2]
    i = 3
    s = 0
    e = 2
    Agg_MFD_True = Agg_MFD_False = Agg_MSA_True = Agg_MSA_False = 0
    for sub in subjectList:
        Agg_MFD_True = Agg_MFD_True + sub[1]
        Agg_MFD_False = Agg_MFD_False + sub[3]
        Agg_MSA_True = Agg_MSA_True + sub[5]
        Agg_MSA_False = Agg_MSA_False + sub[7]
        MSA_means.append(sub[5])
        MSA_means.append(sub[7])
        MSA_std.append(sub[6])
        MSA_std.append(sub[8])
        nTrue = N[sub[0]+'_'+known[0]]
        nFalse = N[sub[0]+'_'+known[1]]
        n = [math.sqrt(nTrue), math.sqrt(nFalse)]
        MSA_std_error = np.divide(MSA_std,n)
        Agg_MSA_err = Agg_MSA_err + MSA_std_error
        pyplot.bar(labels, MSA_means, color = colors[s:e], yerr = MSA_std_error, alpha = 0.5, align = 'center', capsize = 5)
        i,j = labels
        s=s+2
        e=e+2
        labels =[i + 2, j + 2]
        MSA_means = []
        MSA_std = []
        n = []
        MFD_std_error  = []
    pyplot.title("MSA bar charts for different samples(one sample has same color: First - True: Second - False)" )
    pyplot.legend(('S6','S10','S16','S20','S26','S30') , loc = 'upper right')
    pyplot.ylim(top = 16)
    pyplot.xlabel('Labels as in the legend')
    pyplot.ylabel('Mean of MSA points')
    pyplot.show()
    
    # Aggregated graph for MFD 
    labels = [1,2]
    pyplot.bar(labels[0], Agg_MFD_True, color = colors[1], yerr = Agg_MFD_err[0], alpha = 0.5, align = 'center', capsize = 5 )
    pyplot.bar(labels[1], Agg_MFD_False, color = colors[3], yerr = Agg_MFD_err[1], alpha = 0.5, align = 'center', capsize = 5 )
    pyplot.title("MFD bar charts for aggregated samples" )
    pyplot.legend(('Agg_MFD_True','Agg_MFD_False') , loc = 'upper right')
    pyplot.ylim(top = 2)
    pyplot.xlabel('Labels as in the legend')
    pyplot.ylabel('Aggregated Mean of MFD')
    pyplot.show()
    
    # Aggregated graph for MSA
    pyplot.bar(labels[0], Agg_MSA_True, color = colors[1], yerr = Agg_MSA_err[0], alpha = 0.5, align = 'center', capsize = 5 )
    pyplot.bar(labels[1], Agg_MSA_False, color = colors[3], yerr = Agg_MSA_err[1], alpha = 0.5, align = 'center', capsize = 5 )
    pyplot.title("MSA bar charts for aggregated samples" )
    pyplot.legend(('Agg_MSA_True','Agg_MSA_False') , loc = 'upper right')
    pyplot.ylim(top = 49)
    pyplot.xlabel('Labels as in the legend')
    pyplot.ylabel('Aggregated Mean of MSA')
    pyplot.show() 
    
def printOutputToCsv(fileName, data):
    """ Prints data to the output file. """

    outputFile    = open(fileName, "w")
    csvOutputFile = csv.writer(outputFile, delimiter=',')
    for subject in data:
        csvOutputFile.writerow(subject)
    outputFile.close()

def findLengthOfEachSample(centroids):
    """ Finds length of each sample. """

    N = {}
    ids = ['s6_true','s6_false','s10_true','s10_false','s16_true','s16_false','s20_true','s20_false','s26_true','s26_false','s30_true','s30_false']
    for eachID in ids:
        n = 0
        for eachSample in centroids[eachID]:
            n = n + len(eachSample)
        N[eachID] = n
    return N   

# ============================================================================
# ====  MAIN RUTINE START  ===================================================
# ============================================================================

def main(argc, argv):
    """ Main function of the script. """
    
    # =============================
    # ========== TASK 1 ===========    

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
    inputData = loadData()
    dataSet   = dataPreprocessing(inputData)
    inputData = None

    dataSet = convertUnitsToDegree(dataSet)
    centroids = detectCentroids(dataSet)

    # =============================
    # ========== TASK 2 ===========

    subjectList = list()

    subjectList.append(calculateStatisticsOneSubject("s6", centroids["s6_true"], centroids["s6_false"]))    
    subjectList.append(calculateStatisticsOneSubject("s10", centroids["s10_true"], centroids["s10_false"]))
    subjectList.append(calculateStatisticsOneSubject("s16", centroids["s16_true"], centroids["s16_false"]))
    subjectList.append(calculateStatisticsOneSubject("s20", centroids["s20_true"], centroids["s20_false"]))    
    subjectList.append(calculateStatisticsOneSubject("s26", centroids["s26_true"], centroids["s26_false"]))
    subjectList.append(calculateStatisticsOneSubject("s30", centroids["s30_true"], centroids["s30_false"]))

    # =============================
    # ========== TASK 3 ===========

    printOutputToCsv("group10.csv", subjectList)

    # =============================
    # ========== TASK 4 ===========

    N = findLengthOfEachSample(centroids)
    drawGraph(subjectList,N)
   
# ============================================================================
# ====  MAIN RUTINE END  =====================================================
# ============================================================================

if __name__ == "__main__":
	""" Calls the function main()."""
	main(len(sys.argv), sys.argv)
    
# End of file: main.py