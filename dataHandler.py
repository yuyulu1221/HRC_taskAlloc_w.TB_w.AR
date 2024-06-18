import pandas as pd
import numpy as np

id = "final4"

class Vec3:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

## read position
def read_POS(id) -> dict:
	pos_df = pd.read_csv(f"./data/position_{id}.csv")
	Pos = {}
	for _, pos in pos_df.iterrows():
		# Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
		Pos[pos["Name"]] = Vec3(float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"]))
	return Pos

# def cal_dist(pos:dict, p1, p2):
# 	return np.linalg.norm(pos[p1] - pos[p2])

def cal_dist(p1:Vec3, p2:Vec3) -> float:
	tmp = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
	return np.linalg.norm(tmp)

## Read Methods Time Measurement (Therblig process time)
def read_MTM():
	mtm_df = pd.read_excel(f"./data/therblig_process_time.xlsx", index_col=0)
	return mtm_df

## Read Bot Time Measurement Result (Bot process time)
def read_BOTM(id):
	botm_df = pd.read_csv(f"./data/bot_process_time_{id}.csv", index_col=0)
	return botm_df

POS = read_POS(id)
MTM = read_MTM()
BOTM = read_BOTM(id)

# while True:
# 	tmp = input()
# 	p1, p2 = tmp.split()
# 	print(cal_dist(POS, p1, p2))