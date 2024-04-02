from GAScheduling import *
import copy
from therbligHandler import *
import numpy as np

# def read_pos():
# 	pos_df = pd.read_excel("data1.xlsx", sheet_name="Position")
# 	Pos = {}
# 	for idx, pos in pos_df.iterrows():
# 		Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
# 	return Pos

# def read_MTM():
# 	mtm_df = pd.read_excel("data_test.xlsx", sheet_name="Therblig Process Time")
# 	return mtm_df

POS = read_POS()
MTM = read_MTM()

tbh = TBHandler()
tbh.run()

# method to generate job
oht_list_per_job = []
# num_job = len(oht_list_per_job)
# for i, oht in enumerate(tbh.OHT_list):
# 	oht_list_per_job[i % num_job].append(copy.deepcopy(oht))

oht_list_per_job.append(tbh.OHT_list[0:1])
oht_list_per_job.append(tbh.OHT_list[1:3])
# oht_list_per_job.append(tbh.OHT_list[3:4])
# oht_list_per_job.append(tbh.OHT_list[4:5])
# oht_list_per_job.append(tbh.OHT_list[5:7])
 
print(oht_list_per_job)

solver = GASolver(oht_list_per_job)
solver.run()