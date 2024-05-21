import numpy as np

class SSolver:
	def __init__(self, oht_list):
		# Human, Bot, HRC
		self.job_time = [
      		[85, 80, 88],
      		[85, 80, 88],
      		[85, 80, 88],
      		[85, 80, 88],
      		[85, 80, 88],
      		[85, 80, 88],
		]
		self.oht_list = oht_list
	
	def get_job_time(self):
		pass
	
	def cal_makespan(self):
		makespan = 0
		for job in self.job_time:
			makespan += min(job)
		return makespan