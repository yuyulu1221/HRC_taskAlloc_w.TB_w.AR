#%% importing required modules, classes, and functions
from math import inf
from enum import Enum
import pandas as pd
import numpy as np
import copy
from datetime import timedelta
from therbligHandler import *
import dataHandler as dh
import matplotlib.pyplot as plt
import plotly.express as px

class AgentType(Enum):
	LH = 0
	RH = 1
	BOT = 2
 
def read_OHT_relation(oht_list, id):
	ohtr_df = pd.read_csv(f"./data/{id}_oht_relation.csv", index_col=0)
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
	def __init__(self, id, oht_list, pop_size=400, num_iter=20, crossover_rate=0.9, mutation_rate=0.014, rk_mutation_rate=0.014, rk_iter_change_rate=0.66):
		
		self.procedure_id = id
		self.num_repeat = 5
  
		# Get OHT relation
		self.oht_list = read_OHT_relation(oht_list, id)
		self.num_oht = len(oht_list) 
		self.alloc_random_key = [[0.5, 0.5, 0.5] for _ in range(self.num_oht)]
		self.num_agent = 3

		# Hyper-paremeters
		self.pop_size=int(pop_size) 
		self.num_iter=int(num_iter) 
		self.parent_selection_rate = 0.6
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		mutation_selection_rate = 0.2
		self.num_mutation_pos = round(self.num_oht * mutation_selection_rate)
		self.rk_mutation_rate = rk_mutation_rate
		self.rk_iter_change_rate = rk_iter_change_rate
		
		self.pop_list = []
		self.pop_fit_list = []
		self.rk_pop_list = []
		self.alloc_pop_list = []
  
		self.PUN_val = 100000
  
	def test(self):
		pop = [9, 10, 7, 4, 3, 0, 1, 8, 12, 2, 6, 13, 11, 5]
		alloc_pop = [0, 1, 0, 2, 1, 2, 1, 0, 0, 0, 1, 1, 1, 0]
		alloc_pop = [AgentType(ap) for ap in alloc_pop]
 
	def run(self):
		self.init_pop()
		best_list = []
		for it in range(self.num_iter):
			self.Tbest_local = 999999999
			parent, rk_parent = self.selection()
			offspring, rk_offspring, alloc_offspring = self.reproduction(parent, rk_parent)
			self.replacement(offspring, rk_offspring, alloc_offspring)
			best_list.append(self.Tbest_local/10)
			self.progress_bar(it)
		self.cal_makespan(self.pop_best, self.alloc_best, show_result=True)
		self.draw_run_chart(best_list)
   
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
			for alloc_pop in self.alloc_pop_list:
				self.apply_alloc_limit(alloc_pop) 
			self.pop_fit_list.append(self.cal_makespan(self.pop_list[i], self.alloc_pop_list[i]))
   
	def apply_alloc_limit(self, alloc_pop):
		for oht_id in dh.AL:
			alloc_pop[oht_id] = AgentType(dh.AL[oht_id])
 
	def decide_agent(self, rkey) -> int:
		"""
  		Decide agent by random key
		
		Args:
			key (list): Random key for each oht
		Returns:
			AgentType
		"""
		if rkey[0] == rkey[1] and rkey[0] == rkey[2]:
			return np.random.choice((AgentType.LH, AgentType.RH, AgentType.BOT))
		elif rkey[0] == rkey[1] and rkey[0] > rkey[2]:
			return np.random.choice((AgentType.LH, AgentType.RH))
		elif rkey[1] == rkey[2] and rkey[1] > rkey[0]:
			return np.random.choice((AgentType.RH, AgentType.BOT))
		elif rkey[0] == rkey[2] and rkey[2] > rkey[1]:
			return np.random.choice((AgentType.LH, AgentType.BOT))
		else:
			return AgentType(np.argmax(rkey))
   
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
   
	def reproduction(self, parents:list, rk_parents:list) -> list:
		offspring = []
		rk_offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate)):
			## Choose 2 parents from parent list
			i, j = np.random.choice(len(parents), 2, replace = False)
			p0, p1 = parents[i], parents[j]
			rk_p0, rk_p1 = rk_parents[i], rk_parents[j]
   
			offspring.append(self.mask_crossover(p0, p1))
			rk_offspring.append(self.random_key_crossover(rk_p0, rk_p1))
   
		alloc_offspring = [[self.decide_agent(rk) for rk in rk_offspring[idx]] for idx in range(len(rk_offspring))]
		for ao in alloc_offspring:
			self.apply_alloc_limit(ao)

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
		for rk in child:
			if self.rk_mutation_rate >= np.random.rand():
				np.random.shuffle(rk)
		return child

	def random_key_autoreproduction(self, parent) -> list:
		child = []
		for p in parent:
			p_copy = p.copy()
			np.random.shuffle(p_copy)
			child.append(p_copy)
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
			## To avoid repeated searches
			is_searched = [False for _ in range(self.num_oht)]
			## Find oht which has no previous task
			todo_id = find_prev_oht(oht_list[id])
			## Add oht id to output
			output.append(todo_id)
			is_scheduled[todo_id] = True
			## Add bind oht id to output if it exists
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
			self.pop_best = seq_best_local
			self.alloc_best = alloc_best_local
   
	def progress_bar(self, n):
		bar_cnt = (int(((n+1)/self.num_iter)*20))
		space_cnt = 20 - bar_cnt		
		bar = "▇" * bar_cnt + " " * space_cnt
		if n+1 == self.num_iter:
			print(f"\rProgress: [{bar}] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter}, T-best: {self.Tbest}, Alloc: {self.alloc_best[:-1]}")
		else:
			print(f"\rProgress: [{bar}] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter}, T-best: {self.Tbest}, Alloc: {[a.name for a in self.alloc_best[:-1]]}", end="")

	def draw_gantt_chart(self, gantt_dict:dict):
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
				'Agent': ['LH', 'RH', 'BOT'],
				'Resource': [f"OHT{i}" for i in range(self.num_oht - 1)]
			},
			text='Resource',
		)
		fig.update_layout(font=dict(size=36))
		fig.show()
  
	def draw_run_chart(self, best_list:list):
		iterations = [it+1 for it in range(self.num_iter)]
		plt.plot(iterations, best_list)
		plt.title('Run Chart')
		plt.xlabel('iterations')
		plt.ylabel('fitness')
		plt.grid(axis='y', linestyle='--')
		plt.savefig(f'chart/{self.procedure_id}_OHT_run_chart')
		# plt.show()
 
	def cal_makespan(self, pop:list, alloc_pop:list, show_result=False):
		"""
		Returns:
			int: makespan calculated by scheduling
		"""
		ag_time = [0 for _ in range(self.num_agent)]
  
		## Current position for each agent
		cur_pos = ["LH", "RH", "BOT"]

		## Record end time of each OHT
		oht_end_time = [0 for _ in range(self.num_oht)]

		## Special cases: binded OHT is scheduled
		bind_is_scheduled = False
		bind_end_time = 0
  
		gantt_dict = []
		order_list = [[] for _ in range(3)] 
  
		timestamps = [[], [], []]

		for oht_id in pop:
			ag:AgentType = alloc_pop[oht_id]
			oht:OHT = self.oht_list[oht_id]
			process_time = int(oht.get_oht_time(cur_pos[ag.value], ag))
			remain_time = oht.get_bind_remain_time(cur_pos[ag.value], ag)

			if bind_is_scheduled:
				end_time = bind_end_time
				bind_is_scheduled = False

			## For scheduling first binded OHT 
			elif oht.bind != None:
				## Revise alloc in binded task
				bind_ag = alloc_pop[oht.bind.id]
				## Find the end time of current OHT
				seq_time = 0
				if oht.prev:
					seq_time = max(oht_end_time[oht_prev.id] for oht_prev in oht.prev)
				start_time = max(ag_time[ag.value], seq_time)
				start_time = self.revise_start_time(ag, timestamps, start_time, oht.repr_pos)
				end_time = start_time + process_time
				
				## Find the end time of bind OHT
				bind_process_time = int(oht.bind.get_oht_time(cur_pos[bind_ag.value], bind_ag))
				bind_remain_time = oht.bind.get_bind_remain_time(cur_pos[bind_ag.value], bind_ag)
				bind_seq_time = 0
				if oht.bind.prev:
					bind_seq_time = max(oht_end_time[bind_oht_prev.id] for bind_oht_prev in oht.bind.prev)
				bind_start_time = max(ag_time[bind_ag.value], bind_seq_time)
				bind_start_time = self.revise_start_time(bind_ag, timestamps, bind_start_time, oht.repr_pos)
				bind_end_time = max(ag_time[bind_ag.value], bind_start_time) + bind_process_time

				bind_is_scheduled = True

				## Add punishment when using same agent at same time
				same_agent_PUN = 0
				if ag == bind_ag:
					same_agent_PUN = self.PUN_val

				bind_end_time = max(end_time - remain_time, bind_end_time - bind_remain_time) + bind_remain_time + same_agent_PUN
				end_time = max(end_time - remain_time, bind_end_time - bind_remain_time) + remain_time + same_agent_PUN

				start_time = end_time - process_time
				timestamps[ag.value].extend(oht.get_timestamp(ag.name, ag))
				oht.renew_agent_pos(cur_pos, ag)

				bind_start_time = bind_end_time - bind_process_time
				timestamps[bind_ag.value].extend(oht.bind.get_timestamp(bind_ag.name, bind_ag))
				oht.bind.renew_agent_pos(cur_pos, bind_ag)

			## Normal OHT with previous OHT
			elif oht.prev:
				seq_time = 0    
				seq_time = max(oht_end_time[oht_prev.id] for oht_prev in oht.prev)
				start_time = max(ag_time[ag.value], seq_time)
				start_time = self.revise_start_time(ag, timestamps, start_time, oht.repr_pos)
				end_time = start_time + process_time

				timestamps[ag.value].extend(oht.get_timestamp(ag.name, ag))
				oht.renew_agent_pos(cur_pos, ag)

			## Normal OHT without previous OHT
			else:
				start_time = ag_time[ag.value]
				start_time = self.revise_start_time(ag, timestamps, start_time, oht.repr_pos)
				end_time = start_time + process_time
				timestamps[ag.value].extend(oht.get_timestamp(ag.name, ag))
				oht.renew_agent_pos(cur_pos, ag)

			ag_time[ag.value] = end_time
			oht_end_time[oht_id] = end_time
   
			if show_result:
				start_time_delta = str(timedelta(seconds = end_time - process_time)) # convert seconds to hours, minutes and seconds
				end_time_delta = str(timedelta(seconds = end_time))
	
				## Data for Gantt chart
				gantt_dict.append(dict(
					Agent = f'{ag.name}', 
					Start = f'2024-08-01 {str(start_time_delta)}', 
					Finish = f'2024-08-01 {str(end_time_delta)}',
					Resource =f'OHT{oht_id}({self.oht_list[oht_id].type})')
				)
				
				## Data for AR system			
				order_list[ag.value].append(dict(Order = oht.id))

		makespan = max(ag_time)
  
		if show_result:
			print(f"Best OHT sequence: \n-----\t", pop)
			print(f"Best choice of agent: \n-----\t", [a.name for a in alloc_pop])
			print(f"Makespan: ", makespan)
			for a, ord in enumerate(order_list):
				order_df = pd.DataFrame(ord)
				order_df.to_csv(f"./result/{self.procedure_id}_OHT_result_{AgentType(a).name}.csv" ,index=False)
			self.draw_gantt_chart(gantt_dict)
		return makespan

	def revise_start_time(self, ag, timestamps, start_time, pos) -> int:
		if pos == "":
			return start_time

		## When using the left hand, determine the moment when the next right hand is on the right side, and the robotic arm is in front.
		if ag == AgentType.LH:
			for ts in timestamps[AgentType.RH.value]:
				if start_time < ts.time and dh.POS[pos].x > dh.POS[ts.pos].x:
					start_time = ts.time
			for ts in timestamps[AgentType.BOT.value]:
				if start_time < ts.time and dh.POS[pos].z > dh.POS[ts.pos].z:
					start_time = ts.time
		## When using the right hand, determine the moment when the next left hand is on the left side, and the robotic arm is in front.
		if ag == AgentType.RH:
			for ts in timestamps[AgentType.LH.value]:
				if start_time < ts.time and dh.POS[pos].x < dh.POS[ts.pos].x:
					start_time = ts.time
			for ts in timestamps[AgentType.BOT.value]:
				if start_time < ts.time and dh.POS[pos].z > dh.POS[ts.pos].z:
					start_time = ts.time
		## When using the robotic arm, determine the moment when both the left hand and right hand are behind.
		if ag == AgentType.BOT:
			for ts in timestamps[AgentType.LH.value]:
				if start_time < ts.time and dh.POS[pos].z < dh.POS[ts.pos].z:
					start_time = ts.time
			for ts in timestamps[AgentType.RH.value]:
				if start_time < ts.time and dh.POS[pos].z < dh.POS[ts.pos].z:
					start_time = ts.time
		return start_time