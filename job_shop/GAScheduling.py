#%% importing required modules
import pandas as pd
import numpy as np
import time
import copy
import plotly.express as px
import datetime

from sympy import Q

#%% GASolver
class GASolver(object):
	def __init__(self):
		pt_tmp=pd.read_excel("JSP_dataset.xlsx",sheet_name="Processing Time",index_col =[0])
		ms_tmp=pd.read_excel("JSP_dataset.xlsx",sheet_name="Machines Sequence",index_col =[0])

		dfshape=pt_tmp.shape
		self.num_mc=dfshape[1] # number of machines
		self.num_job=dfshape[0] # number of jobs
		self.num_gene= self.num_mc * self.num_job # number of genes in a chromosome

		self.process_t=[list(map(int, pt_tmp.iloc[i])) for i in range(self.num_job)]
		self.machine_seq=[list(map(int,ms_tmp.iloc[i])) for i in range(self.num_job)]

		# Raw input
		self.pop_size=int(input('Please input the size of population: ') or 30) # default value is 30
		self.crossover_rate=float(input('Please input the size of Crossover Rate: ') or 0.8) # default value is 0.8
		self.mutation_rate=float(input('Please input the size of Mutation Rate: ') or 0.2) # default value is 0.2
		mutation_selection_rate=float(input('Please input the mutation selection rate: ') or 0.2)
		self.num_mutation_jobs=round(self.num_gene*mutation_selection_rate)
		self.num_iter=int(input('Please input number of iteration: ') or 100) # default value is 2000
			
		self.pop_list = np.zeros((self.pop_size, self.num_gene))
		self.pop_fit = np.zeros(self.pop_size)
   
		self.makespan_rec = []
		self.start_time = time.time()
  
	def init_pop(self) -> None:
		self.Tbest=999999999999999
		self.best_list, self.best_obj=[],[]
		for i in range(self.pop_size):
			nxm_random_num=list(np.random.permutation(self.num_gene)) # generate a random permutation of 0 to num_job*num_mc-1
			self.pop_list.append(nxm_random_num) # add to the population_list
			for j in range(self.num_gene):
				self.pop_list[i][j] = self.pop_list[i][j] % self.num_job # convert to job number format, every job appears m times
    
	def run(self):
		self.init_pop()
		for n in range(self.num_iter):
			self.Tbest_now = 99999999999
			parent = self.selection()
			offspring = self.twoPtCrossover(parent) # with mutation
			offspring = self.repairment(offspring)
			self.fit()
			self.replacement()
			self.gantt_chart()
   
	def selection(self):
		"""
		roulette wheel approach
		"""
		parent = []
		parent_fit = []
		cumulate_prop = []
		total_fit = 0

		for i in range(self.pop_size):
			self.pop_fit[i] = self.fit(self.pop_list[i])
			total_fit += self.pop_fit[i]

		cumulate_prop.append(self.pop_fit[0])
		for i in range(1, self.pop_size):
			cumulate_prop.append(cumulate_prop[-1] + self.pop_fit[i])
			
		for i in range(0, int(self.pop_size * self.crossover_rate)): 
			for j in range(len(cumulate_prop)):
				select_rand = np.random.rand()
				if select_rand <= cumulate_prop[j]:
					parent.append(copy.deepcopy(self.pop_list[j]))
		
		return parent
   
	def twoPtCrossover(self, parent):
     
		offspring_list = []
		
		for m in range(len(parent)):
			parent_1, parent_2 = np.random.choice(parent, 2, replace=False)
			child_1=parent_1[:]
			child_2=parent_2[:]
			cutpoint=list(np.random.choice(self.num_gene, 2, replace=False))
			cutpoint.sort()
		
			child_1[cutpoint[0]:cutpoint[1]]=parent_2[cutpoint[0]:cutpoint[1]]
			child_2[cutpoint[0]:cutpoint[1]]=parent_1[cutpoint[0]:cutpoint[1]]

			# Mutation
			if self.mutation_rate >= np.random.rand():
				m_chg=list(np.random.choice(self.num_gene, self.num_mutation_jobs, replace=False)) # chooses the position to mutation
				t_value_last=self.offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
				for i in range(self.num_mutation_jobs-1):
					self.offspring_list[m][m_chg[i]]=self.offspring_list[m][m_chg[i+1]] # displacement
				
				self.offspring_list[m][m_chg[self.num_mutation_jobs-1]]=t_value_last # move the value of the first mutation position to the last mutation position
    
	def repairment(self):
		"""
		TODO: 要看懂ㄟ
		"""
		for m in range(self.pop_size):
			job_count={}
			larger,less=[],[] # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
			for i in range(self.num_job):
				if i in self.offspring_list[m]:
					count = self.offspring_list[m].count(i)
					pos = self.offspring_list[m].index(i)
					job_count[i]=[count,pos] # store the above two values to the job_count dictionary
				else:
					count=0
					job_count[i]=[count,0]
				if count > self.num_mc:
					larger.append(i)
				elif count< self.num_mc:
					less.append(i)
					
			for k in range(len(larger)):
				chg_job=larger[k]
				while job_count[chg_job][0] > self.num_mc:
					for d in range(len(less)):
						if job_count[less[d]][0] < self.num_mc:                    
							self.offspring_list[m][job_count[chg_job][1]]=less[d]
							job_count[chg_job][1] = self.offspring_list[m].index(chg_job)
							job_count[chg_job][0] = job_count[chg_job][0]-1
							job_count[less[d]][0] = job_count[less[d]][0]+1                    
						if job_count[chg_job][0]== self.num_mc:
							break     
	
	def mutation(self):
		for m in range(len(self.offspring_list)):
			mutation_prob=np.random.rand()
			if self.mutation_rate >= mutation_prob:
				m_chg=list(np.random.choice(self.num_gene, self.num_mutation_jobs, replace=False)) # chooses the position to mutation
				t_value_last=self.offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
				for i in range(self.num_mutation_jobs-1):
					self.offspring_list[m][m_chg[i]]=self.offspring_list[m][m_chg[i+1]] # displacement
				
				self.offspring_list[m][m_chg[self.num_mutation_jobs-1]]=t_value_last # move the value of the first mutation position to the last mutation position

	def fit(self, pop):
		j_keys=[j for j in range(self.num_job)]
		key_count={key:0 for key in j_keys}
		j_count={key:0 for key in j_keys}
        # 1print(j_count)
		m_keys=[j+1 for j in range(self.num_mc)]
		m_count={key:0 for key in m_keys}
		for i in pop:
			gen_t=int(self.process_t[i][key_count[i]])
			gen_m=int(self.machine_seq[i][key_count[i]])
			j_count[i]=j_count[i]+gen_t
			m_count[gen_m]=m_count[gen_m]+gen_t
			
			if m_count[gen_m]<j_count[i]:
				m_count[gen_m]=j_count[i]
			elif m_count[gen_m]>j_count[i]:
				j_count[i]=m_count[gen_m]
			
			key_count[i]=key_count[i]+1
		makespan=max(j_count.values())
		chrom_fitness = 1/makespan
		return chrom_fitness


  
	def fitnessXX(self):
		total_chromosome=copy.deepcopy(self.parent_list)+copy.deepcopy(self.offspring_list) # parent and offspring chromosomes combination
		chrom_fitness,chrom_fit=[],[]
		total_fitness=0
		for m in range(self.pop_size*2):
			j_keys=[j for j in range(self.num_job)]
			key_count={key:0 for key in j_keys}
			j_count={key:0 for key in j_keys}
			m_keys=[j+1 for j in range(self.num_mc)]
			m_count={key:0 for key in m_keys}
			
			for i in total_chromosome[m]:
				gen_t=int(self.process_t[i][key_count[i]])
				gen_m=int(self.machine_seq[i][key_count[i]])
				j_count[i]=j_count[i]+gen_t
				m_count[gen_m]=m_count[gen_m]+gen_t
				
				if m_count[gen_m]<j_count[i]:
					m_count[gen_m]=j_count[i]
				elif m_count[gen_m]>j_count[i]:
					j_count[i]=m_count[gen_m]
				
				key_count[i]=key_count[i]+1
		
			makespan=max(j_count.values())
			chrom_fitness.append(1/makespan)
			chrom_fit.append(makespan)
			total_fitness=total_fitness+chrom_fitness[m]
		return total_fitness
  
	def replacement(self, pop_fit, pop):
		for i in range(self.pop_size*2):
			if pop_fit[i]< self.Tbest_now:
				self.Tbest_now=pop_fit[i]
				sequence_now=copy.deepcopy(pop[i])
		if self.Tbest_now<=self.Tbest:
			self.Tbest=self.Tbest_now
			self.sequence_best=copy.deepcopy(sequence_now)
   
		self.makespan_rec.append(self.Tbest)
   
	def progress_bar(self, n):
		bar_cnt = (int(((n+1)/self.num_iter)*20))
		space_cnt = 20 - bar_cnt
		print("\rProgress: [" + "█"*bar_cnt + " "*space_cnt + f"] {((n+1)/self.num_iter):.2%} {n+1}/{self.num_iter} T-best = {self.Tbest}", end="")
   
	def gantt_chart(self):
		m_keys=[j+1 for j in range(self.num_mc)]
		j_keys=[j for j in range(self.num_job)]
		key_count={key:0 for key in j_keys}
		j_count={key:0 for key in j_keys}
		m_count={key:0 for key in m_keys}
		j_record={}
		for i in self.sequence_best:
			gen_t=int(self.process_t[i][key_count[i]])
			gen_m=int(self.machine_seq[i][key_count[i]])
			j_count[i]=j_count[i]+gen_t
			m_count[gen_m]=m_count[gen_m]+gen_t
			
			if m_count[gen_m]<j_count[i]:
				m_count[gen_m]=j_count[i]
			elif m_count[gen_m]>j_count[i]:
				j_count[i]=m_count[gen_m]
			
			start_time=str(datetime.timedelta(seconds=j_count[i]-self.process_t[i][key_count[i]])) # convert seconds to hours, minutes and seconds
			
			end_time=str(datetime.timedelta(seconds=j_count[i]))
				
			j_record[(i,gen_m)]=[start_time,end_time]
			
			key_count[i]=key_count[i]+1
			

		df=[]
		for m in m_keys:
			# for j in j_keys:
			#     df.append(dict(Task='Machine %s'%(m), Start='2018-07-14 %s'%(str(j_record[(j,m)][0])), Finish='2018-07-14 %s'%(str(j_record[(j,m)][1])),Resource='Job %s'%(j+1)))
			for j in j_keys:
				df.append(dict(Task=f'Machine {m}', Start=f'2024-03-17 {str(j_record[(j,m)][0])}', Finish=f'2024-03-17 {str(j_record[(j,m)][1])}',Resource=f'Job {j+1}'))

		df = pd.DataFrame(df)

		# fig = px.timeline(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
		# py.iplot(fig, filename='GA_job_shop_scheduling', world_readable=True)

		fig = px.timeline(df, x_start='Start', x_end='Finish', y='Task', color='Resource', title='Job shop Schedule')
		fig.update_yaxes(autorange="reversed")
		fig.show()
  
solver = GASolver()
solver.run()