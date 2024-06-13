from GAScheduling_job import GAJobSolver
from GAScheduling_oht import *
from therbligHandler import *
import optuna

procedure_id = "final3"
num_tbs = 7

tbh = TBHandler(num_tbs=num_tbs, id=procedure_id)
tbh.run()
# print(tbh.job_list)
print(tbh.oht_list)

# test
def test():
	solver = GASolver(procedure_id, tbh.oht_list)
	solver.test()

def oht_simple_run():
	solver = GASolver(procedure_id, tbh.oht_list)
	solver.run()

def oht_optuna_run():
	def GA_objective(trial, id, oht_list):
		param_grid = {
			'pop_size': trial.suggest_int("pop_size", 200, 700, step=50),
			'num_iter': trial.suggest_int("num_iter", 200, 700, step=50),
			'crossover_rate': trial.suggest_float("crossover_rate", 0.6, 1),
			'mutation_rate': trial.suggest_float("mutation_rate", 0.01, 0.025),
			'rk_mutation_rate': trial.suggest_float("rk_mutation_rate", 0.01, 0.025),
			'rk_iter_change_rate': trial.suggest_float("rk_iter_change_rate", 0.4, 0.8)
		}
		solver = GASolver(id, oht_list, **param_grid)
		Tbest = solver.run()
		return Tbest

	study = optuna.create_study(directions=["minimize"])
	study.optimize(lambda trial: GA_objective(trial, procedure_id, tbh.oht_list), n_trials=80, n_jobs=2)

	print('Trial Number: ', study.best_trial.number)
	print('Parameters: ', study.best_trial.params)
	print('Values: ', study.best_trial.value)
	print('para', study.best_trial.user_attrs)
 
def job_simple_run():
	solver = GAJobSolver(procedure_id, tbh.job_list, tbh.oht_list)
	solver.run()
 
def job_optuna_run():
	def GA_objective(trial, id, job_list, oht_list):
		param_grid = {
			'pop_size': trial.suggest_int("pop_size", 200, 700, step=50),
			'num_iter': trial.suggest_int("num_iter", 200, 700, step=50),
			'crossover_rate': trial.suggest_float("crossover_rate", 0.6, 1),
			'mutation_rate': trial.suggest_float("mutation_rate", 0.01, 0.025),
			'rk_mutation_rate': trial.suggest_float("rk_mutation_rate", 0.01, 0.025),
			'rk_iter_change_rate': trial.suggest_float("rk_iter_change_rate", 0.4, 0.8)
		}+98
		solver = GAJobSolver(id, job_list, oht_list, **param_grid)
		Tbest = solver.run()
		return Tbest

	study = optuna.create_study(directions=["minimize"])
	study.optimize(lambda trial: GA_objective(trial, procedure_id, tbh.job_list, tbh.oht_list), n_trials=80, n_jobs=2)

	print('Trial Number: ', study.best_trial.number)
	print('Parameters: ', study.best_trial.params)
	print('Values: ', study.best_trial.value)
	print('para', study.best_trial.user_attrs)

## Run Method
# test()
# oht_simple_run()
# optuna_run()

job_simple_run()
# job_optuna_run()
