import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt



nLines = 3
year = []
day = []
hour = []
minute = []
dst = []
datetime_list = []

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

    booleans_shorttime = (time>start_time)&(time<end_time)

    time_trimmed=time[booleans_shorttime]
    symh_trimmed=symh[booleans_shorttime]

    new_dictionary['times'] = time_trimmed
    new_dictionary['symh'] = symh_trimmed
    return new_dictionary



def count_geomagnetic_storms(dictionary,dst_reference,storm_duration):
    symh = np.array(dictionary['symh'])
    time = np.array(dictionary['times'])

    counter = 0
    storm_found = False
    # Need to save start and end times and count as a storm only if it lasts 12hours
    last_storm_start_time = time[0]
    for i in range(len(symh)):
        if (symh[i] < dst_reference and not(storm_found)):
            storm_start_time = time[i]

            current_storm_duration = (storm_start_time-last_storm_start_time).total_seconds()/3600.
            if (current_storm_duration < storm_duration):
                # print ( current_storm_duration )
                continue
            counter = counter+1

            last_storm_start_time = storm_start_time
            storm_found = True

        elif(storm_found and symh[i] > dst_reference):
            storm_found=False



    # storm_occurance = (symh<dst_reference)
    #
    # storm_times =time[storm_occurance]
    # storm_symh = symh[storm_occurance]
    # dictionary['times'] = storm_times
    # dictionary['symh'] = storm_symh
    # counter = 0
    # storm_found = False
    # for symh_value in dictionary['symh']:
    #     if (symh_value < dst_reference and not(storm_found)):
    #         # print (symh_value)
    #         counter = counter+1
    #         storm_found = True
    #     elif(storm_found and symh_value > dst_reference):
    #         # print ("when turning false", symh_value)
    #         storm_found = False

    return counter, dictionary
# This is the main code

if __name__ == "__main__":
    filename = "omni_min_def_wdjmVKe8ea.lst"
    filename_2003 = "2003_symh_data.lst"
    # start_plot_time = dt.datetime(2013, 3, 17)
    # end_plot_time = dt.datetime(2013, 3, 18)
    start_plot_time = dt.datetime(2003, 10, 28)
    end_plot_time = dt.datetime(2003, 10, 29)
    datetime_dst_dict = read_ascii_file(filename_2003, -1)
    dst_reference = -100
    storm_duration = 12
    number_of_storms, storms_in_2003 = count_geomagnetic_storms(datetime_dst_dict,dst_reference,storm_duration)
    print (number_of_storms)
    # new_dictionary = datetime_dst_dict
    # datetime_dict_toplot = trim_database(datetime_dst_dict, start_plot_time, end_plot_time)


    # time
    # short_period =
    fig,ax = plt.subplots()
    ax.plot(datetime_dst_dict['times'], datetime_dst_dict['symh'], label="OMNI high res")
    ax.set_ylabel("SYMH (nT)")
    ax.set_xlabel("Time")
    ax.set_title("Geomagnetic Storm in 17 March 2013")
    # ax.set_ylim(-100,-150)
    # ax.set_xlim(start_plot_time,end_plot_time)
    ax.legend()
    plt.xticks(rotation=40)
    plt.show()
