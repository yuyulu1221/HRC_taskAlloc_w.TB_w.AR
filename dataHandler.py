import pandas as pd
import numpy as np

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

#%% read OHT relation
def read_OHT_relation(oht_list, id):
	ohtr_df = pd.read_csv(f"./data/oht_relation_{id}.csv", index_col=0)
	print(ohtr_df)
 
	for row_id in range(ohtr_df.shape[0]):
		for col_id in range(ohtr_df.shape[1]):
			if ohtr_df.iloc[row_id][col_id] == -1:
				oht_list[row_id].prev.append(oht_list[col_id])
			elif ohtr_df.iloc[row_id][col_id] == 1:
				oht_list[row_id].next.append(oht_list[col_id])
			elif ohtr_df.iloc[row_id][col_id] == 2:
				oht_list[row_id].bind = oht_list[col_id]
    
	return oht_list