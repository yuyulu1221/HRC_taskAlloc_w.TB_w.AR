import pandas as pd

procedure_id = input("Procedure ID: ")

res_df = pd.read_csv(f"./data/result_{procedure_id}_BOT.csv")
pre_id = ""
with open(f"./data/result_{procedure_id}_BOT.txt", "w") as file:
	for i, res in res_df.iterrows():
		if res["TaskId"] != pre_id:
			file.write(f'{res["TaskId"]}, new List<TaskData>>\n')
   
		file.write('new TaskData { '+ f'Therblig="{res["Name"]}", Position=POS["{res["Position"]}"], Time={res["time"]*0.0036}f' +' },\n')

		pre_id = res["TaskId"]