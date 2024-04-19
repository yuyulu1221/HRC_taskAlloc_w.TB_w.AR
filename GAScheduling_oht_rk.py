#%% importing required modules
from operator import index
import pandas as pd
import numpy as np
import time
import copy
import plotly.express as px
import datetime
from datetime import timedelta

from sympy import Q
import therbligHandler as tbh
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
	return oht_list

#%% OHT structure
class OHTNode:
	def __init__(self, val):
		self.val = val
		self.next = []
		self.prev = []
  
	def add_next(self, n):
		self.next.append(n)
	
	def set_parent(self, p):
		self.prev.append(p)

#%% GASolver
class GASolver():
	def __init__(self, oht_list):
     
		# pt_tmp = pd.read_excel("data_test.xlsx",sheet_name="Processing Time",index_col =[0])
		# ms_tmp = pd.read_excel("data_test.xlsx",sheet_name="Agents Sequence",index_col =[0])

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
  
		self.alloc_random_key = [[0.5, 0.5, 0.5] for _ in range(self.num_oht)]
		self.alloc_res = [0 for _ in range(self.num_oht)]

		self.num_agent = 3

		self.num_gene = self.num_oht * 4 # 1 id + 3 random key

		# Raw input
		self.pop_size=int(input('Please input the size of population: ') or 32) # default: 32
		self.parent_selection_rate=float(input('Please input the size of Parent Selection Rate: ') or 0.5) # default: 0.8
		self.crossover_rate=float(input('Please input the size of Crossover Rate: ') or 0.8) # default: 0.8
		self.mutation_rate=float(input('Please input the size of Mutation Rate: ') or 0.2) # default: 0.2
		mutation_selection_rate=float(input('Please input the mutation selection rate: ') or 0.2)
		self.num_mutation_pos=round(self.num_gene / 4 * mutation_selection_rate)
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
			parent = self.selection()
			offspring = self.twoPtCrossover(parent) # with mutation
			offspring, fit = self.repairment(offspring)
			self.replacement(offspring, fit)
			self.progress_bar(it)
		print("\n")
		# self.gantt_chart()
  
	def test(self):
		self.init_pop()
		for it in range(self.num_iter):
			self.Tbest_now = 999999999
			parent, rk_parent = self.selection()
			offspring = self.maskCrossover(parent) # with mutation
			rk_offspring = self.randomKeyCrossover(rk_parent)
			# print("mask_crossover: ", offspring)
			self.replacement(offspring, rk_offspring)
			self.progress_bar(it)
		print("\n")
  
   
	def init_pop(self) -> None:
		self.Tbest = 999999999
		tmp = list(np.random.permutation(self.num_oht))
		for i in range(self.pop_size):	
			pop = list(np.random.permutation(tmp)) # generate a random permutation of 0 to num_job*num_mc-1
			pop = self.relation_repairment(pop)
			self.pop_list.append(pop) # add to the population_list
			self.rk_pop_list.append([[np.random.normal(0.5, 0.166) for _ in range(3)] for _ in range(self.num_oht)])
			self.pop_fit.append(self.cal_makespan(self.pop_list[i], self.rk_pop_list[i]))
		# print(self.pop_list)

	def relation_repairment(self, oht_seq):
		# print(oht_seq)
		output = []
		oht_list = copy.deepcopy(self.oht_list)
		swap_check = {}
   
		def find_prev_oht(oht: OHT):
			# print("find oht ", oht.id)
			if oht.prev:
				can_choose = []
				for oht_p in oht.prev:
					if oht_p.is_scheduled == False:
						can_choose.append(oht_p)
				if can_choose:
					return find_prev_oht(np.random.choice(can_choose))
				else:
					# print("nothing can choose")
					return oht.id
			else:
				# print("no prev")
				return oht.id
   
		for id in oht_seq:
			# print(f"oht_list: {oht_list}")
			if oht_list[id].is_scheduled == True:
				while swap_check.get(id):
					id = swap_check[id]
			# print("id: ", id)
			if not oht_list[id].prev:
				output.append(id)
				oht_list[id].is_scheduled = True
			else:
				oht_id = find_prev_oht(oht_list[id])
				output.append(oht_id)
				oht_list[oht_id].is_scheduled = True
				if oht_id != id:
					swap_check[oht_id] = id
					# print(str(oht_id) + " -> " + str(id))
				# is_scheduled[oht_todo.id] = 1
		# print(f"output: {output}")
		# input()
		return output	
 
	def decide_agent(self, key) -> int:
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
 
	def cal_makespan(self, pop, rk_pop):
		agent_time = [0 for _ in range(self.num_agent)]
		agent_POS = {
			"LH": self.POS["LH"],
			"RH": self.POS["RH"],
			"BOT": self.POS["BOT"]
		}
		for oht_id in pop:
			agent = self.decide_agent(rk_pop[oht_id])
			self.alloc_res[oht_id] = agent
			oht = self.oht_list[oht_id]
			# print(oht)
			process_time = int(oht.get_oht_time(agent, agent_POS, self.POS, self.MTM))
			if self.oht_list[oht_id]:
				# oht start time should be bigger than every previous oht
				job_time = max(oht_prev.end_time for oht_prev in self.oht_list[oht_id].prev) if self.oht_list[oht_id].prev else 0
				end_time = max(agent_time[agent], job_time) + process_time
			else:
				end_time = agent_time[agent] + process_time
			agent_time[agent] = end_time
			self.oht_list[oht_id].end_time = end_time

		makespan = max(agent_time)
		return makespan
   
	def cal_makespan_job(self, pop_id, pop): # fitness
		# print("pop: ", pop)
		agent_time = [0 for _ in range(self.num_agent)]
		job_time = [0 for _ in range(self.num_job)]
		oht_cnt_per_job = [0 for _ in range(self.num_job)] # next task index for every job which should be executed
		agent_POS = {
			"LH": self.POS["LH"],
			"RH": self.POS["RH"],
			"BOT": self.POS["BOT"]
		}
		for job_id in pop: 
			job_id = int(job_id)
   
			# get the allocated agent
			agent = self.decide_agent(pop_id, job_id, oht_cnt_per_job[job_id])
			self.alloc_per_job[job_id][oht_cnt_per_job[job_id]] = agent
			# print(agent)
   
			# get the task to do
			oht = self.oht_list_per_job[job_id][oht_cnt_per_job[job_id]]
   
			# get process time from composed therbligs
			process_time = int(oht.get_oht_time(agent, agent_POS, self.POS, self.MTM))
			
   			# TODO: Different condition with task sequence
			end_time = max(agent_time[agent], job_time[job_id]) + process_time
			agent_time[agent] = end_time
			job_time[job_id] = end_time
			oht_cnt_per_job[job_id] += 1
   
		makespan = max(agent_time)
		return makespan
   
	def selection(self):
		"""
		roulette wheel approach
		"""
		parent = []
		rk_parent = []
		cumulate_prop = []
		total_fit = 0
		# print(self.pop_list)
		for i in range(self.pop_size):
			self.pop_fit[i] = self.cal_makespan(self.pop_list[i], self.rk_pop_list[i])
			total_fit += self.pop_fit[i]
		# print(self.pop_fit)
		cumulate_prop.append(self.pop_fit[0])
		for i in range(1, self.pop_size):
			cumulate_prop.append(cumulate_prop[-1] + self.pop_fit[i]/total_fit)
		# print(cumulate_prop)
			
		for i in range(0, round(self.pop_size * self.parent_selection_rate)): 
			for j in range(len(cumulate_prop)):
				select_rand = np.random.rand()
				if select_rand <= cumulate_prop[j]:
					parent.append(copy.copy(self.pop_list[j]))
					rk_parent.append(copy.copy(self.rk_pop_list[j]))
		# print(parent)
		return parent, rk_parent
   
	def maskCrossover(self, parents):
		offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate)):
			i, j = np.random.choice(len(parents), 2, replace=False)
			parent0, parent1 = parents[i], parents[j]
			# print("p0: ", parent0)
			# print("p1: ", parent1)
			mask = [np.random.choice([False, True]) for _ in range(self.num_oht)]
			# print("mask: ", mask)
			child = np.arange(self.num_oht)
			is_placed = np.zeros(self.num_oht)
			for i, p in enumerate(parent0):
				if mask[i] == True:
					child[i] = p
					is_placed[p] = 1
			p2_i = 0
			for i in range(self.num_oht):
				if mask[i] == False:
					while p2_i < self.num_oht and is_placed[parent1[p2_i]]:
						p2_i += 1	
					if p2_i >= self.num_oht:
						break
					child[i] = parent1[p2_i]
					is_placed[parent1[p2_i]] = 1
			if self.mutation_rate >= np.random.rand():
				i, j = np.random.choice(self.num_oht, 2, replace=False)
				child[i], child[j] = child[j], child[i]
			child = self.relation_repairment(child)
			# print("child: ", child)
			offspring.append(child)
		return offspring

	def randomKeyCrossover(self, parents):
		offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate)):
			i, j = np.random.choice(len(parents), 2, replace=False)
			parent0, parent1 = parents[i], parents[j]
			child = (np.array(parent0) + np.array(parent1)) / 2
			offspring.append(child)
		return offspring

	def twoPtCrossover(self, parent):
		offspring = []
		for _ in range(round(self.pop_size * self.crossover_rate / 2)):
			p = np.random.choice(len(parent), 2, replace=False)
			parent_1, parent_2 = parent[p[0]], parent[p[1]]
			child = [copy.deepcopy(parent_1), copy.deepcopy(parent_2)]
			cutpoint=list(np.random.choice(self.num_oht, 2, replace=False))
			cutpoint.sort()
		
			child[0][cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
			child[1][cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]

			# Mutation
			for c in child:
				if self.mutation_rate >= np.random.rand():
					mutation_pos=list(np.random.choice(self.num_oht, self.num_mutation_pos, replace=False)) # chooses the position to mutation
					tmp = c[mutation_pos[0]] # save the value which is on the first mutation position
					for i in range(self.num_mutation_pos-1):
						c[mutation_pos[i]] = c[mutation_pos[i+1]] # displacement
					
					c[mutation_pos[self.num_mutation_pos-1]] = tmp # move the value of the first mutation position to the last mutation position
				c = self.show_repairment(c)
				c = self.relation_repairment(c)
				offspring.append(copy.deepcopy(c))
		return offspring
    
	def show_repairment(self, child):
		job_cnt = [0 for _ in range(self.num_oht)]
		insufficient_job = []
		diff_list = []
		for job_id in child:
			job_cnt[job_id] += 1
		for i in range(self.num_job):
			diff = self.num_oht_per_job[i] - job_cnt[i]
			if diff > 0:
				insufficient_job += [i] * diff
			diff_list.append(diff)

		insufficient_job = list(np.random.permutation(insufficient_job))
		for i in range(len(child)):
			# replace insufficient job with insufficient job
			if diff_list[child[i]] < 0:
				diff_list[child[i]] += 1
				child[i] = insufficient_job.pop()
    
	# def repairment(self, offspring):
	# 	"""
	# 	Fix offspring to be feasible solution
	# 	"""
	# 	fit = []
	# 	for child in offspring:
	# 		job_cnt = [0 for _ in range(self.num_job)]
	# 		insufficient_job = []
	# 		diff_list = []
	# 		for job_id in child:
	# 			job_cnt[job_id] += 1
	# 		for i in range(self.num_job):
	# 			diff = self.num_oht_per_job[i] - job_cnt[i]
	# 			if diff > 0:
	# 				insufficient_job += [i] * diff
	# 			diff_list.append(diff)

	# 		insufficient_job = list(np.random.permutation(insufficient_job))
	# 		for i in range(len(child)):
	# 			# replace insufficient job with insufficient job
	# 			if diff_list[child[i]] < 0:
	# 				diff_list[child[i]] += 1
	# 				child[i] = insufficient_job.pop()
     
	# 		# print("child_after: ", child)
	# 		# input()
	# 		fit.append(self.cal_makespan_job(child))
			
	# 	return offspring, fit

	def replacement(self, offspring, rk_offspring):
     
		offspring_fit = []
		for i in range(len(offspring)):
			offspring_fit.append(self.cal_makespan(offspring[i], rk_offspring[i]))
   
		self.pop_list = list(self.pop_list) + offspring
		self.pop_fit = list(self.pop_fit) + offspring_fit

		# Sort
		tmp = sorted(list(zip(self.pop_fit, list(self.pop_list), list(self.rk_pop_list))))
		self.pop_fit, self.pop_list, self.rk_pop_list = zip(*tmp)
		self.pop_list = list(self.pop_list[:self.pop_size])
		self.pop_fit = list(self.pop_fit[:self.pop_size])
		self.rk_pop_list = list(self.rk_pop_list[:self.pop_size])
  
		# print(self.pop_fit)
		
		self.Tbest_now = self.pop_fit[0]
		sequence_now = copy.deepcopy(self.pop_list[0])

		if self.Tbest_now < self.Tbest:
			self.Tbest = self.Tbest_now
			self.sequence_best = copy.deepcopy(sequence_now)
   
		# self.makespan_rec.append(self.Tbest)
   
	def progress_bar(self, n):
		bar_cnt = (int(((n+1)/self.num_iter)*20))
		space_cnt = 20 - bar_cnt		
		bar = "▇"*bar_cnt + " "*space_cnt
		
		print(f"\rProgress: [{bar}] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter}, T-best_now = {self.Tbest_now}, T-best = {self.Tbest}", end="")
   
	def gantt_chart(self):
		agent_time = [0 for _ in range(self.num_agent)]
		job_time = [0 for _ in range(self.num_job)]
		oht_cnt_per_job = [0 for _ in range(self.num_job)] # next task index for every job which should be executed
		oht_dict = {}
		print(self.alloc_per_job_best)
		input()
		for job_id in self.sequence_best: 
      
			job_id = int(job_id)
   
			oht_id_per_job = oht_cnt_per_job[job_id]
			oht = self.oht_list_per_job[job_id][oht_id_per_job]

			agent = self.alloc_per_job_best[job_id][oht_id_per_job]
			
   
			agent_POS = {
				"LH": self.POS["LH"],
				"RH": self.POS["RH"],
				"BOT": self.POS["BOT"],
			}
   
			process_time = int(oht.get_oht_time(agent, agent_POS, self.POS, self.MTM))
			# print("job_id: ", job_id)		
			end_time = max(agent_time[agent], job_time[job_id]) + process_time
			agent_time[agent] = end_time
			job_time[job_id] = end_time

			start_time = str(timedelta(seconds = end_time - process_time)) # convert seconds to hours, minutes and seconds
			# start_time = j_count[i] - self.process_t[i][key_count[i]]
			# print("s: ", start_time)
			
			end_time = str(timedelta(seconds = end_time))
			# end_time = j_count[i]
			# print("e:", end_time)
				
			oht_dict[(job_id, oht_id_per_job, agent)] = [start_time, end_time]
			# print(f"[{job_id}, {oht_id_per_job}, {agent}] = [{start_time}, {end_time}]")
			
			oht_cnt_per_job[job_id] += 1
			
		tmp = []
		for job_id, oht_list in enumerate(self.oht_list_per_job):
			for oht_id_per_job in range(len(oht_list)):
				agent = self.alloc_per_job_best[job_id][oht_id_per_job]
				tmp.append(dict(
        			Task = f'{AGENT[agent]}', 
           			Start = f'2024-07-14 {(str(oht_dict[(job_id, oht_id_per_job, agent)][0]))}', 
              		Finish = f'2024-07-14 {(str(oht_dict[(job_id, oht_id_per_job, agent)][1]))}',
                	Resource =f'Job{job_id}')
               	)
			# for j in j_keys:
				# df.append(dict(Task=f'Machine {m}', Start=j_record[(j,m)][0], Finish=j_record[(j,m)][1]*1000, Resource=f'Job {j+1}'))
		
		df = pd.DataFrame(tmp)

		# fig = px.timeline(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
		# py.iplot(fig, filename='GA_job_shop_scheduling', world_readable=True)

		fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Resource', title='Job shop Schedule')
		fig.update_yaxes(autorange="reversed")
		fig.show()
  
# solver = GASolver([1,0,2])
# solver.run()