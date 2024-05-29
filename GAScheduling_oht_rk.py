#%% importing required modules
from math import inf
import pandas as pd
import numpy as np
import copy
import plotly.express as px
from datetime import timedelta

from therbligHandler import *
# from InfoHandler import *

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
			if ohtr_df.iloc[row_id, col_id] == -1:
				oht_list[row_id].prev.append(oht_list[col_id])
			elif ohtr_df.iloc[row_id, col_id] == 1:
				oht_list[row_id].next.append(oht_list[col_id])
			elif ohtr_df.iloc[row_id, col_id] == 2:
				oht_list[row_id].bind = oht_list[col_id]
    
	return oht_list

#%% GASolver
class GASolver():
	def __init__(self, id, oht_list, pop_size=100, num_iter=120, crossover_rate=0.7, mutation_rate=0.01):
		
		self.procedure_id = id
  		# Get position dict -> str: np.array_1x3
		self.POS = read_POS(id)
		# Get MTM dataframe
		self.MTM = read_MTM()
		# Get OHT relation
		self.oht_list = read_OHT_relation(oht_list, id)

		## Check oht graph
		# for oht in self.oht_list:
		# 	print("---")
		# 	print(f"OHT{oht.id}")
		# 	print(f"prev: {[oht_p.id for oht_p in oht.prev]}")

		self.num_oht = len(oht_list)
  
		self.alloc_random_key = [[0.5, 0.5, 0.5] for _ in range(self.num_oht)]
		self.alloc_res = [0 for _ in range(self.num_oht)]

		self.num_agent = 3

		# Hyper-paremeters
		self.pop_size=int(pop_size) 
		self.num_iter=int(num_iter) 
		self.parent_selection_rate = 0.6
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		mutation_selection_rate = 0.2
		self.num_mutation_pos = round(self.num_oht * mutation_selection_rate)
		
		self.pop_list = []
		self.pop_fit_list = []
		self.rk_pop_list = []
		self.alloc_pop_list = []
  
		self.PUN_val = 100000
  
	def run(self):
		self.init_pop()
		for it in range(self.num_iter):
			self.Tbest_local = 999999999
			parent, rk_parent = self.selection()
			offspring, rk_offspring, alloc_offspring = self.crossover(parent, rk_parent)
			self.replacement(offspring, rk_offspring, alloc_offspring)
			self.progress_bar(it)
		print("\n")
		self.show_result()
		return self.Tbest
   
	def init_pop(self) -> None:
		self.Tbest = 999999999
		tmp =  [i for i in range(self.num_oht)]
	
		## Init pop_list, rk_pop_list, and pop_fit
		for i in range(self.pop_size):	
			pop = list(np.random.permutation(tmp))
			pop = self.relation_repairment(pop)
			self.pop_list.append(pop)
			rk_pop = [[np.random.normal(0.5, 0.166) for _ in range(3)] for _ in range(self.num_oht)]
			self.rk_pop_list.append(rk_pop)
			self.alloc_pop_list.append([self.decide_agent(rk) for rk in rk_pop])
			self.pop_fit_list.append(self.cal_makespan(self.pop_list[i], self.alloc_pop_list[i]))
	
	def decide_agent(self, key) -> int:
		"""
  		Decide agent by random key
		
		Args:
			key (list): Random key for each oht
		Returns:
			int: Agent id
		"""
		if key[0] == key[1] and key[0] == key[2]:
			return np.random.randint(0, 3)
		elif key[0] == key[1] and key[0] > key[2]:
			return np.random.randint(0, 2)
		elif key[1] == key[2] and key[1] > key[0]:
			return np.random.randint(1, 3)
		elif key[0] == key[2] and key[2] > key[1]:
			return np.random.choice([0, 2])
		else:
			return np.argmax(key)
   
	def selection(self) -> tuple[list, list]:
		"""
		Roulette wheel approach

		Returns:
			tuple: parent and random_key_parent
		"""
		parent = []
		rk_parent = []
		cumulate_prop = []
		total_fit = 0
		
		## Renew pop_fit and calculate total fit for roulette wheel approach
		for i in range(self.pop_size):
			self.pop_fit_list[i] = self.cal_makespan(self.pop_list[i], self.alloc_pop_list[i])
			total_fit += 1 / self.pop_fit_list[i]
		
		## Calculate cumulative propability
		cumulate_prop.append(self.pop_fit_list[0] / total_fit)
		for i in range(1, self.pop_size):
			cumulate_prop.append(cumulate_prop[i-1] + self.pop_fit_list[i] / total_fit)
		
		## Generate parent and rk_parent list
		for i in range(0, round(self.pop_size * self.parent_selection_rate)): 
			for j in range(len(cumulate_prop)):
				select_rand = np.random.rand()
				if select_rand <= cumulate_prop[j]:
					parent.append(copy.copy(self.pop_list[j]))
					rk_parent.append(copy.copy(self.rk_pop_list[j]))

		return parent, rk_parent
   
	def crossover(self, parents:list, rk_parents:list) -> list:
		offspring = []
		rk_offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate)):
			## choose 2 parents from parent list
			i, j = np.random.choice(len(parents), 2, replace = False)
			p0, p1 = parents[i], parents[j]
			rk_p0, rk_p1 = rk_parents[i], rk_parents[j]
   
			offspring.append(self.mask_crossover(p0, p1))
			rk_offspring.append(self.random_key_crossover(rk_p0, rk_p1))
		alloc_offspring = [[self.decide_agent(rk) for rk in rk_offspring[idx]] for idx in range(len(rk_offspring))]

		return offspring, rk_offspring, alloc_offspring
  
	def mask_crossover(self, parent0, parent1) -> list:
		""" 
		Args:
			parents (list): choice parents
		Returns:
			list: offspring list
		"""
		## True for parent0; False for parent1
		mask = [np.random.choice([False, True]) for _ in range(self.num_oht)]
		child = [-1 for _ in range(self.num_oht)]
		is_placed = [False for _ in range(self.num_oht)]

		## Using mask to fill in the value of parent0 into child
		for i, p in enumerate(parent0):
			if mask[i] == True:
				child[i] = p
				is_placed[p] = True

		## Using p1_idx to find next index that can be filled in
		p1_idx = 0
		for i in range(self.num_oht):
			if mask[i] == False:
				while p1_idx < self.num_oht and is_placed[parent1[p1_idx]]:
					p1_idx += 1	
				if p1_idx >= self.num_oht:
					break
				child[i] = parent1[p1_idx]
				is_placed[parent1[p1_idx]] = True
	
		## Sequence mutation: choose 2 position to swap
		if self.mutation_rate >= np.random.rand():
			i, j = np.random.choice(self.num_oht, 2, replace=False)
			child[i], child[j] = child[j], child[i]
		child = self.relation_repairment(child)
   
		return child

	def random_key_crossover(self, parent0, parent1) -> list:

		child = (np.array(parent0) + np.array(parent1)) / 2

		## Random key mutation: choose 2 position to swap
		if self.mutation_rate >= np.random.rand():
			child = [c + np.random.normal() / 10 for c in child]

		return child
    
	def relation_repairment(self, oht_seq) -> list:
		"""
		Maintain OHT sequence

		Args:
			oht_seq (list): original OHT sequence
		Returns:
			list: repaired OHT sequence
		"""
		output = []
		oht_list = copy.deepcopy(self.oht_list)
		is_scheduled = [False for _ in range(self.num_oht)]
		swap = {}
   
		def find_prev_oht(oht: OHT):
			if is_searched[oht.id]:
				return oht.id
			is_searched[oht.id] = True
			if oht.prev:
				can_choose = set()
				for oht_p in oht.prev:
					if is_scheduled[oht_p.id] == False:
						can_choose.add(oht_p)
				if can_choose:
					return find_prev_oht(np.random.choice(list(can_choose)))
				else:
					return find_bind_oht(oht)
			else:
				return find_bind_oht(oht)

		def find_bind_oht(oht: OHT):
			if oht.bind == None:
				return oht.id
			else:
				return find_prev_oht(oht.bind)
   
		for id in oht_seq:

			## Replace current id with unused id
			while swap.get(id):
				id = swap.pop(id)

			## Skip it if the oht is scheduled
			if is_scheduled[id] == True:
				continue
			
			## To avoid repeated searches '''
			is_searched = [False for _ in range(self.num_oht)]

			## Find oht which has no previous task
			todo_id = find_prev_oht(oht_list[id])
			
			## add oht id to output
			output.append(todo_id)
			is_scheduled[todo_id] = True
   
			## add bind oht id to output if it exists
			if oht_list[todo_id].bind != None:
				output.append(oht_list[todo_id].bind.id)
				is_scheduled[oht_list[todo_id].bind.id] = True

			## Record the unused id
			if todo_id != id:
				swap[todo_id] = id
 
		return output	

	def replacement(self, offspring, rk_offspring, alloc_offspring) -> None:
		"""
		Replace worse pop by better offspring		
		"""
		offspring_fit = []
		for i in range(len(offspring)):
			offspring_fit.append(self.cal_makespan(offspring[i], alloc_offspring[i]))
   
		self.pop_list = list(self.pop_list) + offspring
		self.pop_fit_list = list(self.pop_fit_list) + offspring_fit
		self.rk_pop_list = list(self.rk_pop_list) + rk_offspring
		self.alloc_pop_list = list(self.alloc_pop_list) + alloc_offspring

		## Sort by pop fit
		tmp = sorted(list(zip(self.pop_fit_list, list(self.pop_list), list(self.rk_pop_list), list(self.alloc_pop_list))), key=lambda x:x[0])
		self.pop_fit_list, self.pop_list, self.rk_pop_list, self.alloc_pop_list = zip(*tmp)
		self.pop_list = list(self.pop_list[:self.pop_size])
		self.pop_fit_list = list(self.pop_fit_list[:self.pop_size])
		self.rk_pop_list = list(self.rk_pop_list[:self.pop_size])
		self.alloc_pop_list = list(self.alloc_pop_list[:self.pop_size])
		
		## Update local best
		self.Tbest_local = self.pop_fit_list[0]
		seq_best_local = self.pop_list[0]
		alloc_best_local = self.alloc_pop_list[0]

		## Update global best
		if self.Tbest_local < self.Tbest:
			self.Tbest = self.Tbest_local
			self.seq_best = seq_best_local
			self.alloc_best = alloc_best_local
   
	def progress_bar(self, n):
		bar_cnt = (int(((n+1)/self.num_iter)*20))
		space_cnt = 20 - bar_cnt		
		bar = "▇" * bar_cnt + " " * space_cnt
		if n+1 == self.num_iter:
			print(f"\rProgress: [{bar}] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter}, T-best_local = {self.Tbest_local}, T-best = {self.Tbest}")
		else:
			print(f"\rProgress: [{bar}] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter}, T-best_local = {self.Tbest_local}, T-best = {self.Tbest}", end="")
   
	def cal_makespan(self, pop:list, alloc_pop:list):
		"""
		Returns:
			int: makespan calculated by scheduling
		"""
		agent_time = [0 for _ in range(self.num_agent)]
		agent_oht_id = [[] for _ in range(self.num_agent)]
  
		## Current position for each agent
		agent_POS = ["LH", "RH", "BOT"]

		## Record end time of each OHT
		oht_end_time = [0 for _ in range(self.num_oht)]

		## Special cases: binded OHT is scheduled
		bind_is_scheduled = False
		bind_end_time = 0
  
		timestamps = [[], [], []]

		for oht_id in pop:
			agent = alloc_pop[oht_id]
			oht:OHT = self.oht_list[oht_id]
			process_time = int(oht.get_oht_time(agent_POS, agent, self.POS, self.MTM))

			if bind_is_scheduled:
				end_time = bind_end_time
				bind_is_scheduled = False

			## For scheduling first binded OHT 
			elif self.oht_list[oht_id].bind != None:

				## Find the end time of current OHT
				job_time = 0
				if self.oht_list[oht_id].prev:
					job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				end_time = max(agent_time[agent], job_time) + process_time
				
				## Find the end time of bind OHT
				bind_agent = alloc_pop[self.oht_list[oht_id].bind.id]
				bind_process_time = int(oht.bind.get_oht_time(agent_POS, bind_agent, self.POS, self.MTM))
				bind_job_time = 0
				if self.oht_list[oht_id].bind.prev:
					bind_job_time = max(oht_end_time[bind_oht_prev.id] for bind_oht_prev in self.oht_list[oht_id].bind.prev)
				bind_end_time = max(agent_time[bind_agent], bind_job_time) + bind_process_time

				bind_is_scheduled = True

				## Add punishment when using same agent at same time
				same_agent_PUN = 0
				if agent == bind_agent:
					same_agent_PUN = self.PUN_val

				bind_end_time = max(end_time, bind_end_time) + same_agent_PUN
				end_time = max(end_time, bind_end_time) + same_agent_PUN

				start_time = end_time - process_time
				tmp = oht.get_timestamp(agent_POS, agent, self.POS, self.MTM)
				for t, pos in tmp:
					timestamps[agent].append((start_time + t, pos))
				oht.renew_agent_pos(agent_POS, agent, self.POS)

				bind_start_time = bind_end_time - bind_process_time
				tmp = oht.get_timestamp(agent_POS, bind_agent, self.POS, self.MTM)
				for t, pos in tmp:
					timestamps[bind_agent].append((bind_start_time + t, pos))
				oht.bind.renew_agent_pos(agent_POS, bind_agent, self.POS)

			## Normal OHT with previous OHT
			elif self.oht_list[oht_id].prev:
				job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				start_time = max(agent_time[agent], job_time)
				end_time = start_time + process_time

				tmp = oht.get_timestamp(agent_POS, agent, self.POS, self.MTM)
				for t, pos in tmp:
					timestamps[agent].append((start_time + t, pos))
				oht.renew_agent_pos(agent_POS, agent, self.POS)

			## Normal OHT without previous OHT
			else:
				start_time = agent_time[agent]
				end_time = start_time + process_time
	
				tmp = oht.get_timestamp(agent_POS, agent, self.POS, self.MTM)
				for t, pos in tmp:
					timestamps[agent].append((start_time + t, pos))
				oht.renew_agent_pos(agent_POS, agent, self.POS)

			agent_oht_id[agent].append(oht_id)
			agent_time[agent] = end_time
			oht_end_time[oht_id] = end_time

		
		
		makespan = max(agent_time) + self.interference_PUN(timestamps)
		## Verify interference punishment
		# if makespan < self.PUN_val:
		# 	print(pop)
		# 	print(makespan)
		# 	print(timestamps[0])
		# 	print(timestamps[1])
		# 	print(timestamps[2])
		# 	input()

		return makespan

	def interference_PUN(self, timestamps):
     	## Handle interference problem
		i, j, k = 0, 0, 0
		lh_now, rh_now, bot_now = self.POS['LH'], self.POS['RH'], self.POS['BOT']
		pun = 0
  
		## Compare the x-coord of LH and RH; compare the z_coord of robot and hands
		while i < len(timestamps[0]) or j < len(timestamps[1]) or k < len(timestamps[2]):
			## Check interference
			if lh_now[0] > rh_now[0] \
   			or lh_now[2] > bot_now[2] \
      		or rh_now[2] > bot_now[2]:
				pun = self.PUN_val
				break
			## Renew agent position
			if i >= len(timestamps[0]):
				if j >= len(timestamps[1]):
					bot_now = timestamps[2][k][1]
					k += 1
				elif k >= len(timestamps[2]):
					rh_now = timestamps[1][j][1]
					j += 1
				else:
					if timestamps[1][j][0] < timestamps[2][k][0]:
						rh_now = timestamps[1][j][1]
						j += 1 
					elif timestamps[1][j][0] > timestamps[2][k][0]:
						bot_now = timestamps[2][k][1]
						k += 1
					else:
						rh_now = timestamps[1][j][1]
						j += 1 
						bot_now = timestamps[2][k][1]
						k += 1
			elif j >= len(timestamps[1]):
				if k >= len(timestamps[2]):	
					lh_now = timestamps[0][i][1]
					i += 1
				else:
					if timestamps[0][i][0] < timestamps[2][k][0]:
						lh_now = timestamps[0][i][1]
						i += 1 
					elif timestamps[0][i][0] > timestamps[2][k][0]:
						bot_now = timestamps[2][k][1]
						k += 1 
					else:
						lh_now = timestamps[0][i][1]
						i += 1 
						bot_now = timestamps[2][k][1]
						k += 1 
			elif k >= len(timestamps[2]):
				if timestamps[0][i][0] < timestamps[1][j][0]:
					lh_now = timestamps[0][i][1]
					i += 1 
				elif timestamps[0][i][0] < timestamps[1][j][0]:
					rh_now = timestamps[1][j][1]
					j += 1 
				else:
					lh_now = timestamps[0][i][1]
					i += 1 
					rh_now = timestamps[1][j][1]
					j += 1 
			else:
				if timestamps[0][i][0] < min(timestamps[1][j][0], timestamps[2][k][0]):
					lh_now = timestamps[0][i][1]
					i += 1
				elif timestamps[1][j][0] < min(timestamps[0][i][0], timestamps[2][k][0]):
					rh_now = timestamps[1][j][1]
					j += 1
				elif timestamps[2][k][0] < min(timestamps[0][i][0], timestamps[1][j][0]):
					bot_now = timestamps[2][k][1]
					k += 1
				else:
					lh_now = timestamps[0][i][1]
					i += 1
					rh_now = timestamps[1][j][1]
					j += 1
		return pun
   
	def show_result(self):
     
		agent_time = [0 for _ in range(self.num_agent)]
		agent_POS = ["LH", "RH", "BOT"]
  
		gantt_dict = []
		path_dict = [[] for _ in range(3)] 
  
		oht_end_time = [0 for _ in range(self.num_oht)]
		bind_is_scheduled = False
		bind_end_time = 0

		print("\n")
		print(f"Best fit: \n-----\t", self.Tbest)
		print(f"Best OHT sequence: \n-----\t", self.seq_best[:-1])
		print(f"Best choice of agent: \n-----\t", [AGENT[ag] for ag in self.alloc_best])
		print(self.pop_fit_list)
  
		if self.Tbest >= self.PUN_val:
			print("No valid solution!")
			return

		for oht_id in self.seq_best[:-1]: ## not showing END node
      
			## Use already calculated data when scheduling second binded OHT
			agent = self.alloc_best[oht_id]
			oht:OHT = self.oht_list[oht_id]
			process_time = int(oht.get_oht_time(agent_POS, agent, self.POS, self.MTM))
			oht.renew_agent_pos(agent_POS, agent, self.POS)

			## Same as cal_makespan
			if bind_is_scheduled:
				end_time = bind_end_time
				bind_is_scheduled = False
    
			## For scheduling first binded OHT 
			elif self.oht_list[oht_id].bind != None:
				## Find the end time of current OHT
				job_time = 0
				if self.oht_list[oht_id].prev:
					job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				end_time = max(agent_time[agent], job_time) + process_time
    
				## Find the end time of bind OHT
				bind_agent = self.alloc_best[self.oht_list[oht_id].bind.id]
				bind_process_time = int(oht.bind.get_oht_time(agent_POS, bind_agent, self.POS, self.MTM))
				# oht.bind.renew_agent_pos(agent_POS, agent, self.POS)
				bind_job_time = 0
				if self.oht_list[oht_id].bind.prev:
					bind_job_time = max(oht_end_time[bind_oht_prev.id] for bind_oht_prev in self.oht_list[oht_id].bind.prev)
				bind_is_scheduled = True
				bind_end_time = max(agent_time[bind_agent], bind_job_time) + bind_process_time

				end_time = max(end_time, bind_end_time)
				bind_end_time = max(end_time, bind_end_time)

			## Normal OHT with previous OHT
			elif self.oht_list[oht_id].prev:
				## oht start time should be bigger than every previous oht
				job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				end_time = max(agent_time[agent], job_time) + process_time
    
			## Normal OHT without previous OHT
			else:
				end_time = agent_time[agent] + process_time
    
			agent_time[agent] = end_time
			oht_end_time[oht_id] = end_time
   
			start_time_delta = str(timedelta(seconds = end_time - process_time)) # convert seconds to hours, minutes and seconds
			end_time_delta = str(timedelta(seconds = end_time))
			gantt_dict.append(dict(
				Agent = f'{AGENT[agent]}', 
				Start = f'2024-05-27 {(str(start_time_delta))}', 
				Finish = f'2024-05-27 {(str(end_time_delta))}',
				Resource =f'OHT{oht_id}')
            	)
			
			prefix_time = float(end_time - process_time)
			for tb in self.oht_list[oht_id].flat():
				path_dict[agent].append(dict(
					TaskId = oht.id,
					Name = tb.name,
					Start = prefix_time,
					Position = tb.To,
					time = tb.time
				))
				prefix_time += tb.time

		for a, pathd in enumerate(path_dict):
			path_df = pd.DataFrame(pathd)
			path_df.to_csv(f"./data/result_{self.procedure_id}_{AGENT[a]}.csv" ,index=False)
  
		## Draw gantt chart
		gantt_df = pd.DataFrame(gantt_dict)
		fig = px.timeline(
      		gantt_df,
			x_start='Start', 
			x_end='Finish', 
			y='Agent', 
			color='Resource', 
			title='Schedule', 
			color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.Pastel,
			category_orders={
				'Agent': ['BOT', 'RH', 'LH'],
				'Resource': [f"OHT{i}" for i in range(self.num_oht - 1)]
			},
			text='Resource'
   		)
		fig.update_yaxes(autorange="reversed")
		fig.show()
  
# solver = GASolver([1,0,2])
# solver.run()