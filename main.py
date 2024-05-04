from GAScheduling_oht_rk import *
from therbligHandler import *
import optuna

tbh = TBHandler()
tbh.run()

# Simple run
def simple_run():
	solver = GASolver(tbh.OHT_list)
	solver.run()

def optuna_run():
	def GA_objective(trial, oht_list):
		param_grid = {
			'pop_size': trial.suggest_int("pop_size", 20, 300, step=40),
			'num_iter': trial.suggest_int("num_iter", 50, 500, step=50),
			'crossover_rate': trial.suggest_float("crossover_rate", 0.6, 0.9),
			'mutation_rate': trial.suggest_float("mutation_rate", 0.01, 0.025)
		}
		solver = GASolver(oht_list, **param_grid)
		Tbest = solver.run()
		return Tbest

	study = optuna.create_study(directions=["minimize"])
	study.optimize(lambda trial: GA_objective(trial, tbh.OHT_list), n_trials=160, n_jobs=4)

	print('Trial Number: ', study.best_trial.number)
	print('Parameters: ', study.best_trial.params)
	print('Values: ', study.best_trial.value)

## Run Method
# simple_run()
optuna_run()
