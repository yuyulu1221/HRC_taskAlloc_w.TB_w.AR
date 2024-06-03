from numpy import NaN
import pandas as pd

procedure_id = input("Procedure ID: ")

pos_df = pd.read_csv(f"./data/position_{procedure_id}.csv")
with open("./data/position.txt", "w") as file:
	for _, pos in pos_df.iterrows():
		file.write(f'POS.Add("{pos["Name"]}", new Vector3({pos["x_coord"]/100}f, {pos["y_coord"]/100}f, {pos["z_coord"]/100}f));\n')
  
res_df = pd.read_csv(f"./data/result_{procedure_id}_BOT.csv")
pre_id = ""
with open(f"./data/result_{procedure_id}_BOT.txt", "w") as file:
	for i, res in res_df.iterrows():
		if res["TaskId"] != pre_id:
			file.write(f'{res["TaskId"]}, new List<TaskData>>\n')
   
		file.write('new TaskData { '+ f'Therblig="{res["Name"]}", Position=POS["{res["Position"]}"], Time={res["time"]*0.0036}f' +' },\n')

		pre_id = res["TaskId"]