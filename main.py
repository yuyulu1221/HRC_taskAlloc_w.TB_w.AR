from GAScheduling_oht_rk import *
from therbligHandler import *

tbh = TBHandler()
tbh.run()

solver = GASolver(tbh.OHT_list)
solver.run()