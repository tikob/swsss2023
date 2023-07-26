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


"""
This block opens the file, reads the lines and splits them.
It creates datetie objects from the given data.
And saves datetimes as a list.
"""
with open("omni_min_def_wdjmVKe8ea.lst") as f:
    for line in f:
        tmp = line.split()
        dst.append(int(tmp[4]))
        datetime1 = dt.datetime(int(tmp[0]), 1, 1, int(tmp[2]), int(tmp[3]))+dt.timedelta(days=int(tmp[1])-1)
        datetime_list.append(datetime1)
    # print (dathetime_list)

"""
Plotting of the data with the label and rotated X-Axis for times.
"""
fig,ax = plt.subplots()
ax.plot(datetime_list, dst, label="OMNI high res")
ax.set_ylabel("SYMH (nT)")
ax.set_xlabel("Time")
ax.set_title("Geomagnetic Storm in 17 March 2013")
ax.legend()
plt.xticks(rotation=40)
plt.show()
