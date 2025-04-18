from GAScheduling_task import GATaskSolver
from GAScheduling_oht import *
from therbligHandler import *
import optuna

procedure_id = "final"
num_tbs = 7

tbh = TBHandler(num_tbs=num_tbs, id=procedure_id)
tbh.run()
for i, oht in enumerate(tbh.oht_list):
	print(f'OHT{i}\t: {oht}')

def test_oht():
	solver = GASolver(procedure_id, tbh.oht_list)
	solver.test()
 
def test_task():
	solver = GATaskSolver(procedure_id, tbh.task_list, tbh.oht_list)
	solver.test()

def oht_simple_run():
	solver = GASolver(procedure_id, tbh.oht_list)
	solver.run()

def oht_optuna_run():
	def GA_objective(trial, id, oht_list):
		param_grid = {
			'pop_size': trial.suggest_int("pop_size", 100, 500, step=50),
			'num_iter': trial.suggest_int("num_iter", 100, 500, step=50),
			'crossover_rate': trial.suggest_float("crossover_rate", 0.6, 1),
			'mutation_rate': trial.suggest_float("mutation_rate", 0.01, 0.025),
			'rk_mutation_rate': trial.suggest_float("rk_mutation_rate", 0.01, 0.025),
			'rk_iter_change_rate': trial.suggest_float("rk_iter_change_rate", 0.4, 0.8)
		}
		solver = GASolver(id, oht_list, **param_grid)
		Tbest = solver.run()
		return Tbest

	study = optuna.create_study(directions=["minimize"])
	study.optimize(lambda trial: GA_objective(trial, procedure_id, tbh.oht_list), n_trials=160, n_jobs=4)

	print('Trial Number: ', study.best_trial.number)
	print('Parameters: ', study.best_trial.params)
	print('Values: ', study.best_trial.value)
	print('para', study.best_trial.user_attrs)
 
def task_simple_run():
	solver = GATaskSolver(procedure_id, tbh.task_list, tbh.oht_list)
	solver.run()
 
def task_optuna_run():
	def GA_objective(trial, id, task_list, oht_list):
		param_grid = {
			'pop_size': trial.suggest_int("pop_size", 100, 500, step=50),
			'num_iter': trial.suggest_int("num_iter", 100, 500, step=50),
			'crossover_rate': trial.suggest_float("crossover_rate", 0.6, 1),
			'mutation_rate': trial.suggest_float("mutation_rate", 0.01, 0.025),
		}
		solver = GATaskSolver(id, task_list, oht_list, **param_grid)
		Tbest = solver.run()
		return Tbest

	study = optuna.create_study(directions=["minimize"])
	study.optimize(lambda trial: GA_objective(trial, procedure_id, tbh.task_list, tbh.oht_list), n_trials=160, n_jobs=4)

	print('Trial Number: ', study.best_trial.number)
	print('Parameters: ', study.best_trial.params)
	print('Values: ', study.best_trial.value)
	print('para', study.best_trial.user_attrs)


while True:
	cmd = input("Scheduling Type: ")
	if cmd == 'oht':
		oht_simple_run()
	elif cmd == 'task':
		task_simple_run()
	elif cmd == 'testoht':
		test_oht()
	elif cmd == 'testtask':
		test_task()
	## Optuna
	elif cmd == 'optoht':
		oht_optuna_run()
	elif cmd == 'opttask':
		task_optuna_run()
	elif cmd == 'bye':
		print('Bye~')
		break
	elif cmd == '--help' or cmd == '-h':
		print('Commands:\n\
	- oht: oht model\n\
	- task: traditional model\n\
	- testoht: test for one-handed task model\n\
	- testtask: test for traditional model\n\
	- optoht: Optuna module for hyperparameter optimization of the oht model\n\
	- opttask: Optuna module for hyperparameter optimization of the traditional model\n\
	- bye: exit the program\n\
      ')
	else:
		print('Invalid cmd. Type in "-h" for help.')