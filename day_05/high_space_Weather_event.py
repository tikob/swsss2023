"""
    This module creates a new dataset (dictionary) for the high space weather event.
    It gives the start time, the end time and the dataset (dictionary) for 10 hours around this event.

    Usage:
        from high_space_Weather_event import event_interval
        dst_dataset, start_time, end_time = event_interval(filename, data_resolution)
    Inputs:
        filename (string): name of the file
        data_resolution (int): the resolution of the data
    Input examples:
        filename = "omni_min_def_rCgJiRBSvP.lst"
        data_resolution = 1 (for 1m omni data)
    Outputs:
        Start_time (datetime_object): the start of the space weather event
        end_time (datetime_object): the end of the space weather event
        dataset (dictionary): the dictionary of the data between the given dates in 1h format

"""
__author__ = 'Tinatin Baratashvili'
__email__ = 'tinatin.baratashvili@kuleuven.be'



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

def skip_condition(line):
    """
        This function skips the header in the file if it exists.

        If the first character of the line is not a difit it returns False
        If the first character of the line is it returns  True

        ATTENTION: this does not check every character,
        so if the first one is digit and the others strings, it will not work.
        This is because negative numbers are not digits.

    """

    if (not(line[0].isdigit())):
        return False
    return True

def read_ascii_file(filename,index):
    """
    This function returns cthe given index from the data file - filename.
    Inputs:
        filename (string): OMNI data file
        index (int): index of the variable to plot in the file
    Returns:
        Dictionary of datetime objects and the requested index.

    """
    dictionary_datetime_index = {}
    datetimes = []
    symh = []
    with open(filename) as f:
        for line in f:
            if (skip_condition(line)):
                tmp = line.split()
                datetime1 = dt.datetime(int(tmp[0]), 1, 1, int(tmp[2]), int(tmp[3]))+dt.timedelta(days=int(tmp[1])-1)
                datetimes.append(datetime1)
                symh.append(int(tmp[index]))
        dictionary_datetime_index = {'times': datetimes,'symh': symh}

    return dictionary_datetime_index

def trim_database(dictionary, start_time, end_time):
    """
        This functions trims the dictionary between the start and end dates.
        Inputs:
            dictionary (dictionary type): has keys {'times', 'symh'}
            start_time (datetime object): the start time for trimming the dataset
            end_time (datetime object): the end time for trimming the dataset
        Retuns:
            Dictionary that has trimmed dataset in it between start_date and end_date.
    """
    new_dictionary = {}
    time = np.array(dictionary['times'])
    symh = np.array(dictionary['symh'])
    # In order to also include the end time, we add 1 hor to end_time
    # end_time = end_time + dt.timedelta(hours=1)
    print ("in tim database:  ", start_time, end_time)
    booleans_shorttime = (time>start_time)&(time<end_time)

    time_trimmed=time[booleans_shorttime]
    symh_trimmed=symh[booleans_shorttime]

    new_dictionary['times'] = time_trimmed
    new_dictionary['symh'] = symh_trimmed
    print (time_trimmed[0], time_trimmed[-1])
    # dst_data = generate_dst_1h_res(new_dictionary,data_resolution)
    return new_dictionary


def find_minimum_dst(dataset_dictionary):
    """
        This function calculates the minimum dst value
        and the corresponding time for the minimum dst value
        in the iven dataset.
        Inputs:
            dataset_dictionary (dictionary) : dataset for the data with keys 'times' and 'symh'
        Returns:
            min_dst_time (datetime format): time for the minimum dst value
            min_dst (float): the minimum dst value
    """
    min_dst_time = 0
    dst_threshold = -100.0
    min_dst = 0
    times = dataset_dictionary['times']
    symh = dataset_dictionary['symh']
    for j in range(len(times)):
        if symh[j] < min_dst:
            min_dst = symh[j]
            min_dst_time = times[j]
    min_dst_time = min_dst_time - dt.timedelta(minutes=1)
    start_estimate = min_dst_time - dt.timedelta(hours=24)
    end_estimate = min_dst_time + dt.timedelta(hours=24)

    exact_start_time = times[0]
    exact_end_time = times[-1]

    for i in range(len(times)):
        if (times[i] > start_estimate and times[i] < end_estimate):
            if (symh[i] <= dst_threshold):
                exact_start_time = times[i]
                print (exact_start_time, symh[i])
                break
    for i in range(len(times)):
        if (times[i] > min_dst_time and times[i] < end_estimate):
            if(symh[i] >= dst_threshold):
                exact_end_time = times[i]
                break

    print ("print exact_times before 5h ",exact_start_time, exact_end_time)
    exact_start_time = exact_start_time - dt.timedelta(hours=5)
    exact_end_time = exact_end_time + dt.timedelta(hours=5)
    print ("print exact_times after 5h ",exact_start_time, exact_end_time)
    return exact_start_time, exact_end_time, min_dst_time, min_dst

def generate_dst_1h_res(dataset_dictionary,data_resolution):
    """
        This function converts high res(1m) data to low res(1h) data.
        Inputs:
            dataset_dictionary (dictionary) : dataset for the data with keys 'times' and 'symh' with 1m resolution data
        Outputs:
            low_Res_dictionary (dictionary) : dataset for the data with keys 'times' and 'symh' with 1h resolution data

    """
    times = dataset_dictionary['times']
    symh = dataset_dictionary['symh']
    low_res_dictionary = {}
    times_lowres = []
    dst_lowres = []
    final_res = 60
    # print (int(final_res/data_resolution), len(times), len(times)/60)
    times_lowres = [times[int(final_res/data_resolution)*i] for i in range(int(len(times)/60))]
    symh_lowres = [symh[int(final_res/data_resolution)*i] for i in range(int(len(symh)/60))]
    low_res_dictionary['times'] = times_lowres
    low_res_dictionary['symh'] = symh_lowres
    # print (low_res_dictionary['times'])
    return low_res_dictionary

def event_interval(filename, data_resolution):
    """
        This function finds the 10 hour time interval around the high space weather event.
        Inputs:
            filename (string): the name of the file where we are searching for the space weather event.
        Returns:
            event_start_time (datetime object): when the space weather event starts
            event_end_time (datetime_object): when the space weather event ends
            new_dataset (dictionary): new dataset with 1h resolution data between the start and end times
    """

    index_dst = -1
    start_time = 0
    end_time = 0
    dataset_dictionary = read_ascii_file(filename, index_dst)
    low_res_data = generate_dst_1h_res(dataset_dictionary, data_resolution)
    exact_start_time, exact_end_time, min_dst_time, min_dst = find_minimum_dst(dataset_dictionary)
    start_time = min_dst_time - dt.timedelta(hours=24)
    end_time = min_dst_time + dt.timedelta(hours=24)
    new_dataset = trim_database(dataset_dictionary, exact_start_time, exact_end_time)

    return  exact_start_time, exact_end_time, new_dataset



if __name__ == "__main__":
    filename = "omni_min_def_rCgJiRBSvP.lst"
    data_resolution = 1
    event_start_time, event_end_time, dataset_around_storm = event_interval(filename,data_resolution)
    storm_start = event_start_time + dt.timedelta(hours=5)
    storm_end = event_end_time - dt.timedelta(hours=5)
    storm_duration = (storm_end-storm_start).total_seconds()/3600.0
    print ("storm duration: ", storm_duration)
    print (storm_start, event_start_time, storm_end, event_end_time)
    fig,ax = plt.subplots()
    ax.plot(dataset_around_storm['times'], dataset_around_storm['symh'],linewidth=2, label="OMNI 1h res")
    ax.plot((storm_start,storm_start), (min(dataset_around_storm['symh']), max(dataset_around_storm['symh'])), 'r--', linewidth=2, label = "The start of the storm: "+str(storm_start))
    ax.plot((storm_end,storm_end), (min(dataset_around_storm['symh']), max(dataset_around_storm['symh'])), 'k--',linewidth=2, label = "The end of the storm: "+str(storm_end))
    ax.set_ylabel("SYMH (nT)", fontsize=18)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_title("The Strong Space Weather event in 2002", fontsize=18)
    ax.legend(fontsize=18)
    plt.xticks(rotation=40)
    plt.show()

    # print (dataset_around_storm,event_start_time, event_end_time)
