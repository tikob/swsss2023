from high_space_Weather_event import event_interval
import datetime as dt


filename = "omni_min_def_rCgJiRBSvP.lst"
data_resolution = 1
event_start_time, event_end_time, dataset_around_storm = event_interval(filename,data_resolution)


print (event_start_time, event_end_time)
