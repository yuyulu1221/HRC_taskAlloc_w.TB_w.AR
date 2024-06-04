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

#%% read MTM
def read_MTM():
	mtm_df = pd.read_excel(f"./data/therblig_process_time.xlsx", index_col=0)
	return mtm_df

POS = read_POS(id)
MTM = read_MTM()