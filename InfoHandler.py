import pandas as pd
import numpy as np

class Position():
	def __init__(self):
		pos_df = pd.read_excel("data1.xlsx", sheet_name="Position")
		self.Pos = {}
		for idx, pos in pos_df.iterrows():
			self.Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
		# print(self.Pos)
   
	def __repr__(self):
		return str(self.Pos)

class MTM():
	def __init__(self):
		