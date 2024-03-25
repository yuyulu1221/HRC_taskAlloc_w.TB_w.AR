import pandas as pd
import numpy as np
import plotly.express as px
import datetime
    

j_record = {
	(0, 0): [str(datetime.timedelta(seconds=0)), str(datetime.timedelta(seconds=1))],
	(0, 1): [str(datetime.timedelta(seconds=4)), str(datetime.timedelta(seconds=7))],
	(1, 0): [str(datetime.timedelta(seconds=3)), str(datetime.timedelta(seconds=5))],
	(1, 1): [str(datetime.timedelta(seconds=0)), str(datetime.timedelta(seconds=3))],
}

df=[]
for m in range(2):
	# for j in j_keys:
	#     df.append(dict(Task='Machine %s'%(m), Start='2018-07-14 %s'%(str(j_record[(j,m)][0])), Finish='2018-07-14 %s'%(str(j_record[(j,m)][1])),Resource='Job %s'%(j+1)))
	for j in range(2):
		df.append(dict(Task=f'Agent {m}', Start=f'2024-03-17 {str(j_record[(j,m)][0])}', Finish=f'2024-03-17 {str(j_record[(j,m)][1])}',Resource=f"Agent {j}"))

df = pd.DataFrame(df)

# fig = px.timeline(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
# py.iplot(fig, filename='GA_job_shop_scheduling', world_readable=True)

fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Resource', title='Job shop Schedule')
fig.update_yaxes(autorange="reversed")
fig.show()