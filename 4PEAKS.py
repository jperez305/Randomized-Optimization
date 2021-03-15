# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:02:58 2021

@author: Joey
"""

import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose.fitness import FourPeaks
from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, classification_report
from mlrose.opt_probs import  DiscreteOpt
import numpy as np
import matplotlib.pyplot as plt
import mlrose 
import random 

import time


random_seed_list = [10 * i for i in range(5)] 

fp_objective = FourPeaks()
problem = DiscreteOpt(length = 100, fitness_fn = fp_objective)

# OPTIMIZATION STEP

# RHC
start = time.time()
rhc_fitness = []
for seed in random_seed_list:
    best_state, best_fitness, obj_curve = random_hill_climb(problem, max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
    rhc_fitness.append(obj_curve)

print("--- %s seconds ---" % (time.time() - start))
    
plt.plot(np.arange(1,5001), np.mean(np.array(rhc_fitness), axis = 0))
plt.title("RHC ALGORITHM: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")




# SIMULATED ALGORITHMS
#sa_fitness = []
#
#start = time.time()
#for i in [10, 100, 250, 500]:
#    sa_fitness = []
#    for seed in random_seed_list:
#        best_state, best_fitness, obj_curve = simulated_annealing(problem, schedule= mlrose.GeomDecay(init_temp = i) ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
#    sa_fitness.append(obj_curve)
#    print(np.mean(np.array(sa_fitness)))
#print(np.mean(np.array(sa_fitness)))

# SIMULATED ANNEALING
sa_fitness = []
start = time.time()
for seed in random_seed_list:
    best_state, best_fitness, obj_curve = simulated_annealing(problem,schedule= mlrose.GeomDecay(init_temp = 10) ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
    sa_fitness.append(obj_curve)

print("--- %s seconds ---" % (time.time() - start))
    
plt.plot(np.arange(1,5001), np.mean(np.array(sa_fitness), axis = 0))
plt.title("SIMULATED ANNEALING: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")




# GENETIC ALGORITHMS
#final_check =  []
#start = time.time()
#for i in [ 0.20, 0.50, 0.80]:
#    for j in [100, 250, 500]:
#        ga_fitness = []
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = genetic_alg(problem, pop_size = j, mutation_prob = i ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
#        ga_fitness.append(obj_curve)
#        print(np.mean(np.array(ga_fitness)))
#print(np.mean(np.array(ga_fitness)))


ga_fitness = []
start = time.time()
for seed in random_seed_list:
    best_state, best_fitness, obj_curve = genetic_alg(problem, pop_size = 1000, mutation_prob = 0.20 ,max_attempts = 500, max_iters = 500, curve = True, random_state = seed)
    ga_fitness.append(obj_curve)

    

print("--- %s seconds ---" % (time.time() - start))
    
plt.plot(np.arange(1,501), np.mean(np.array(ga_fitness), axis = 0) )
plt.title("GENETIC ALGORITHMS: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")



# MIMIC ALGORITHMS
#mimic_fitness = []
#start = time.time()
#for i in [0.10, 0.30, 0.40]:
#    for j in [100,200, 500]:
#        mimic_fitness = [] = []
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = mimic(problem, pop_size = j, keep_pct = i ,max_attempts = 200, max_iters = 200, curve = True, random_state = seed)
#        mimic_fitness.append(obj_curve)
#        print(np.mean(np.array(mimic_fitness)))
#print(np.mean(np.array(mimic_fitness)))


# MIMIC
mimic_fitness = []
start = time.time()
for seed in random_seed_list:
    best_state, best_fitness, obj_curve = mimic(problem,pop_size = 500 , keep_pct= 0.1,max_attempts = 200, max_iters = 200, curve = True, random_state = seed, fast_mimic= True)
    mimic_fitness.append(obj_curve)
print("--- %s seconds ---" % (time.time() - start))


plt.plot(np.arange(1,201), np.mean(np.array(mimic_fitness), axis = 0) )
plt.title("MIMIC ALGORITHMS: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")

plt.plot(np.arange(1,5001), np.mean(np.array(rhc_fitness), axis = 0), label = "Random Hill Climbing")
plt.plot(np.arange(1,5001), np.mean(np.array(sa_fitness), axis = 0), label = "Simulated Annealing")
plt.plot(np.arange(1,501), np.mean(np.array(ga_fitness), axis = 0) ,label = "GA")
plt.plot(np.arange(1,201), np.mean(np.array(mimic_fitness), axis = 0), label = "MIMIC")
plt.legend()
plt.title("4 Peaks: Iterations vs Objective")
plt.show()
