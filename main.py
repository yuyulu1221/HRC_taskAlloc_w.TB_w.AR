from GAScheduling import *
import copy
from therbligHandler import *
import numpy as np

tbh = TBHandler()
tbh.run()

# method to generate job
oht_list_per_job = [[], [], []]
num_job = len(oht_list_per_job)
for i, oht in enumerate(tbh.OHT_list):
	oht_list_per_job[i % num_job].append(copy.deepcopy(oht))
 
print(oht_list_per_job)

solver = GASolver(oht_list_per_job)
solver.run()