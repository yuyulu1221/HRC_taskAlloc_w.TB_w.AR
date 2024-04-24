from GAScheduling_oht_rk import *
import copy
from therbligHandler import *
import numpy as np

tbh = TBHandler()
tbh.run()


# method to generate job
oht_list_per_job = []
# num_job = len(oht_list_per_job)
# for i, oht in enumerate(tbh.OHT_list):
# 	oht_list_per_job[i % num_job].append(copy.deepcopy(oht))


# oht_list_per_job.append(tbh.OHT_list)
# oht_list_per_job.append(tbh.OHT_list[4:5])
# oht_list_per_job.append(tbh.OHT_list[2:])
 
# print(oht_list_per_job)
solver = GASolver(tbh.OHT_list)

solver.run()