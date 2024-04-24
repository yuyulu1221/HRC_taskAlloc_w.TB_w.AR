#%% importing required modules
from multiprocessing import process
from operator import index
import pandas as pd
import numpy as np
import time
import copy
import plotly.express as px
from datetime import timedelta

from pyparsing import col

from therbligHandler import *
# from InfoHandler import *

#%% read position
def read_POS():
	pos_df = pd.read_excel("data.xlsx", sheet_name="Position")
	Pos = {}
	for idx, pos in pos_df.iterrows():
		Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
	return Pos

#%% read MTM
def read_MTM():
	mtm_df = pd.read_excel("data.xlsx", sheet_name="Therblig Process Time", index_col=0)
	return mtm_df

#%% read OHT relation
def read_OHT_relation(oht_list):
	ohtr_df = pd.read_excel("data.xlsx", sheet_name="OHT Relation", index_col=0)
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

#%% GASolver
class GASolver():
	def __init__(self, oht_list):
		
  		# Get position dict -> str: np.array_1x3
		self.POS = read_POS()
		# Get MTM dataframe
		self.MTM = read_MTM()
		# Get OHT relation
		# self.OHT_in_edge = read_OHT_relation()
		self.oht_list = read_OHT_relation(oht_list)
		for oht in self.oht_list:
			print("---")
			print(oht.prev)
			print(oht.next)

		self.num_oht = len(oht_list)
		# self.num_seq = 0
  
		self.alloc_random_key = [[0.5, 0.5, 0.5] for _ in range(self.num_oht)]
		self.alloc_res = [0 for _ in range(self.num_oht)]

		self.num_agent = 3

		self.num_gene = self.num_oht * 4 # 1 id + 3 random key

		# Raw input
		self.pop_size=int(input('Please input the size of population: ') or 64) 
		# self.pop_size = 64
		# self.parent_selection_rate=float(input('Please input the size of Parent Selection Rate: ') or 0.5) # default: 0.8
		self.parent_selection_rate = 0.8
		# self.crossover_rate=float(input('Please input the size of Crossover Rate: ') or 0.8) # default: 0.8
		self.crossover_rate = 0.8
		# self.mutation_rate=float(input('Please input the size of Mutation Rate: ') or 0.2) # default: 0.2
		self.mutation_rate = 0.2
		mutation_selection_rate=float(input('Please input the mutation selection rate: ') or 0.2)
		self.num_mutation_pos = round(self.num_gene / 4 * mutation_selection_rate)
		self.num_iter=int(input('Please input number of iteration: ') or 200) # default: 2000
			
		self.pop_list = []
		self.pop_fit = []
		self.rk_pop_list = []
   
		self.makespan_rec = []
		self.start_time = time.time() # ? 
  
	def run(self):
		self.init_pop()
		for it in range(self.num_iter):
			self.Tbest_now = 999999999
			parent, rk_parent = self.selection()
			offspring = self.maskCrossover(parent)
			rk_offspring = self.randomKeyCrossover(rk_parent)
			self.replacement(offspring, rk_offspring)
			self.progress_bar(it)
		print("\n")
		self.gantt_chart()
   
	def init_pop(self) -> None:
		self.Tbest = 999999999
		tmp =  [i for i in range(self.num_oht)]
	
		## Init pop_list, rk_pop_list, and pop_fit
		for i in range(self.pop_size):	
			pop = list(np.random.permutation(tmp))
			pop = self.relation_repairment(pop)
			self.pop_list.append(pop)
			self.rk_pop_list.append([[np.random.normal(0.5, 0.166) for _ in range(3)] for _ in range(self.num_oht)])
			self.pop_fit.append(self.cal_makespan(self.pop_list[i], self.rk_pop_list[i]))
	
	def decide_agent(self, key) -> int:
		"""
  		Decide agent by random key
		
		Args:
			key (list): Random key for each oht
		Returns:
			int: Agent id
		"""
		if key[0] == key[1] == key[2]:
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
			self.pop_fit[i] = self.cal_makespan(self.pop_list[i], self.rk_pop_list[i])
			total_fit += self.pop_fit[i]
		
		## Calculate cumulative propability
		cumulate_prop.append(self.pop_fit[0] / total_fit)
		for i in range(1, self.pop_size):
			cumulate_prop.append(cumulate_prop[-1] + self.pop_fit[i]/total_fit)
		
		## Generate parent and rk_parent list
		for i in range(0, round(self.pop_size * self.parent_selection_rate)): 
			for j in range(len(cumulate_prop)):
				select_rand = np.random.rand()
				if select_rand <= cumulate_prop[j]:
					parent.append(copy.copy(self.pop_list[j]))
					rk_parent.append(copy.copy(self.rk_pop_list[j]))

		return parent, rk_parent
   
	def maskCrossover(self, parents:list) -> list:
		""" 
		Args:
			parents (list): choice parents
		Returns:
			list: offspring list
		"""
		offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate)):
      
			## choose 2 parents from parent list
			i, j = np.random.choice(len(parents), 2, replace = False)
			parent0, parent1 = parents[i], parents[j]
   
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
     
			## Mutation: choose 2 position to swap
			if self.mutation_rate >= np.random.rand():
				i, j = np.random.choice(self.num_oht, 2, replace=False)
				child[i], child[j] = child[j], child[i]
			child = self.relation_repairment(child)
			offspring.append(child)
	
		return offspring

	def randomKeyCrossover(self, parents) -> list:
		""" 
		Args:
			parents (list): choice parents
		Returns:
			list: random key offspring list
		"""
		offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate)):
			i, j = np.random.choice(len(parents), 2, replace=False)
			parent0, parent1 = parents[i], parents[j]
			child = (np.array(parent0) + np.array(parent1)) / 2
			offspring.append(child)
		return offspring
    
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
			if oht.bind == -1:
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
			if oht_list[todo_id].bind != -1:
				output.append(oht_list[todo_id].bind.id)
				is_scheduled[oht_list[todo_id].bind.id] = True

			## Record the unused id
			if todo_id != id:
				swap[todo_id] = id
 
		return output	

	def replacement(self, offspring, rk_offspring) -> None:
		"""
		Replace worse pop by better offspring		
		"""
		offspring_fit = []
		for i in range(len(offspring)):
			offspring_fit.append(self.cal_makespan(offspring[i], rk_offspring[i]))
   
		self.pop_list = list(self.pop_list) + offspring
		self.pop_fit = list(self.pop_fit) + offspring_fit

		## Sort by pop fit
		tmp = sorted(list(zip(self.pop_fit, list(self.pop_list), list(self.rk_pop_list))))
		self.pop_fit, self.pop_list, self.rk_pop_list = zip(*tmp)
		self.pop_list = list(self.pop_list[:self.pop_size])
		self.pop_fit = list(self.pop_fit[:self.pop_size])
		self.rk_pop_list = list(self.rk_pop_list[:self.pop_size])
		
		## Update local best
		self.Tbest_now = self.pop_fit[0]
		sequence_now = copy.deepcopy(self.pop_list[0])
		random_key_now = copy.deepcopy(self.rk_pop_list[0])

		## Update global best
		if self.Tbest_now < self.Tbest:
			self.Tbest = self.Tbest_now
			self.sequence_best = copy.deepcopy(sequence_now)
			self.random_key_best = copy.deepcopy(random_key_now)
   
	def progress_bar(self, n):
		bar_cnt = (int(((n+1)/self.num_iter)*20))
		space_cnt = 20 - bar_cnt		
		bar = "▇"*bar_cnt + " "*space_cnt
		print(f"\rProgress: [{bar}] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter}, T-best_now = {self.Tbest_now}, T-best = {self.Tbest}", end="")
   
	def cal_makespan(self, pop:list, rk_pop:list):
		agent_time = [0 for _ in range(self.num_agent)]
		agent_oht_id = [[] for _ in range(self.num_agent)]
		agent_POS = {
			"LH": self.POS["LH"],
			"RH": self.POS["RH"],
			"BOT": self.POS["BOT"]
		}
		oht_end_time = np.zeros(self.num_oht)
		bind_is_scheduled = False
		bind_start_time = 0
		for oht_id in pop:
			agent = self.decide_agent(rk_pop[oht_id])
			self.alloc_res[oht_id] = agent
			oht = self.oht_list[oht_id]
			process_time = int(oht.get_oht_time(agent, agent_POS, self.POS, self.MTM))

			if bind_is_scheduled:
				bind_is_scheduled = False
				end_time = bind_start_time + process_time
			elif self.oht_list[oht_id].bind != -1:
				bind_is_scheduled = True
				job_time = 0
				if self.oht_list[oht_id].prev:
					job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				bind_job_time = 0
				if self.oht_list[oht_id].bind.prev:
					bind_job_time = max(oht_end_time[bind_oht_prev.id] for bind_oht_prev in self.oht_list[oht_id].bind.prev)
				bind_agent = self.decide_agent(rk_pop[self.oht_list[oht_id].bind.id])
				bind_start_time = max(agent_time[agent], agent_time[bind_agent], job_time, bind_job_time)
				punishment = 0
				if agent == bind_agent:
					punishment = 1000000
				end_time = bind_start_time + process_time + punishment
			elif self.oht_list[oht_id].prev:
				# oht start time should be bigger than every previous oht
				job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				end_time = max(agent_time[agent], job_time) + process_time
			else:
				end_time = agent_time[agent] + process_time

			agent_oht_id[agent].append(oht_id)
			agent_time[agent] = end_time
			oht_end_time[oht_id] = end_time

		makespan = max(agent_time)
  
		# lower the oht random key of the highest makespan
		for oht_id in agent_oht_id[np.argmax(agent_time)]:
			rk_pop[oht_id][agent] -= 0.01

		return makespan
   
	def gantt_chart(self):
		agent_time = [0 for _ in range(self.num_agent)]
		agent_POS = {
			"LH": self.POS["LH"],
			"RH": self.POS["RH"],
			"BOT": self.POS["BOT"]
		}
		tmp = []
		oht_end_time = [0 for _ in range(self.num_oht)]
		bind_is_scheduled = False
		bind_start_time = 0
  
		print(f"OHT sequence: ", self.sequence_best[:-1])
		print(f"Random key: ", [AGENT[self.decide_agent(rkb)] for rkb in self.random_key_best][:-1])

		for oht_id in self.sequence_best[:-1]: # not showing END node
			agent = self.decide_agent(self.random_key_best[oht_id])
			oht = self.oht_list[oht_id]
			process_time = int(oht.get_oht_time(agent, agent_POS, self.POS, self.MTM))
			if bind_is_scheduled:
				bind_is_scheduled = False
				end_time = bind_start_time + process_time
			elif self.oht_list[oht_id].bind != -1:
				bind_is_scheduled = True
				job_time = 0
				if self.oht_list[oht_id].prev:
					job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				bind_job_time = 0
				bind_agent_time = 0
				if self.oht_list[oht_id].bind.prev:
					bind_job_time = max(oht_end_time[bind_oht_prev.id] for bind_oht_prev in self.oht_list[oht_id].bind.prev)
				bind_agent = self.decide_agent(self.random_key_best[self.oht_list[oht_id].bind.id])
				bind_agent_time = agent_time[bind_agent]
				bind_start_time = max(agent_time[agent], bind_agent_time, job_time, bind_job_time)
				end_time = bind_start_time + process_time
			elif self.oht_list[oht_id].prev:
				# oht start time should be bigger than every previous oht
				job_time = max(oht_end_time[oht_prev.id] for oht_prev in self.oht_list[oht_id].prev)
				end_time = max(agent_time[agent], job_time) + process_time
			else:
				end_time = agent_time[agent] + process_time
			agent_time[agent] = end_time
			oht_end_time[oht_id] = end_time
   
			start_time = str(timedelta(seconds = end_time - process_time)) # convert seconds to hours, minutes and seconds
			end_time = str(timedelta(seconds = end_time))
			tmp.append(dict(
        			Task = f'{AGENT[agent]}', 
           			Start = f'2024-04-24 {(str(start_time))}', 
            		Finish = f'2024-04-24 {(str(end_time))}',
            		Resource =f'OHT{oht_id}')
            	)
		
		df = pd.DataFrame(tmp)

		fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Resource', title='Schedule')
		fig.update_yaxes(autorange="reversed")
		fig.show()
  
# solver = GASolver([1,0,2])
# solver.run()