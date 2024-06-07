import pandas as pd
import numpy as np

id = "final2"

#%% read position
def read_POS(id):
	pos_df = pd.read_csv(f"./data/position_{id}.csv")
	Pos = {}
	for _, pos in pos_df.iterrows():
		Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
	return Pos

def cal_dist(pos:dict, p1, p2):
	return np.linalg.norm(pos[p1] - pos[p2])

#%% read MTM
def read_MTM():
	mtm_df = pd.read_excel(f"./data/therblig_process_time.xlsx", index_col=0)
	return mtm_df

POS = read_POS(id)
MTM = read_MTM()

# while True:
# 	tmp = input()
# 	p1, p2 = tmp.split()
# 	print(cal_dist(POS, p1, p2))