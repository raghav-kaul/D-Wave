# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import sys
for i in range(int(sys.argv[1]),int(sys.argv[2])):
    os.system(f'nohup python3 bqm_simulated_annealing.py {i} >Output_arrays/v2_sweeps_output{i}.log &')