# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:50:18 2021

@author: Joey
"""


import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose.fitness import FlipFlop
from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.opt_probs import TSPOpt, DiscreteOpt
import numpy as np
import matplotlib.pyplot as plt
import mlrose 
import time


random_seed_list = [10 * i for i in range(5)]

ff_objective = FlipFlop()



problem = DiscreteOpt(length = 100, fitness_fn = ff_objective)


# OPTIMIZATION STEP

# RHC
rhc_fitness = []
start = time.time()
for seed in random_seed_list:
    best_state, best_fitness, obj_curve = random_hill_climb(problem, max_attempts = 500, max_iters = 500, curve = True, random_state = seed)
    rhc_fitness.append(obj_curve)
print("--- %s seconds ---" % (time.time() - start))

    
plt.plot(np.arange(1,501), np.mean(np.array(rhc_fitness), axis = 0))
plt.title("Random Hill Climbing: Iterations vs Objective")

plt.xlabel("Iterations")
plt.ylabel("Value")



# SIMULATED ALGORITHMS
#sa_fitness = []
#
#start = time.time()
#for i in [10,50, 100, 200, 500, 1000]:
#    for j in [0.1, 0.2, 0.3, 0.5, 0.8]:
#        sa_fitness = []
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = simulated_annealing(problem, schedule= mlrose.GeomDecay(init_temp = i, exp_const = j) ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
#        sa_fitness.append(obj_curve)
#        print(np.mean(np.array(sa_fitness)))
#print(np.mean(np.array(sa_fitness)))


start = time.time()
# SIMULATED ANNEALING
sa_fitness = []
for seed in random_seed_list:
    best_state, best_fitness, obj_curve = simulated_annealing(problem,schedule= mlrose.ExpDecay(init_temp = 10, exp_const = 0.8) ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
    sa_fitness.append(obj_curve)

print("--- %s seconds ---" % (time.time() - start))

plt.plot(np.arange(1,5001), np.mean(np.array(sa_fitness), axis = 0))
plt.title("SIMULATED ANNEALING: Iterations vs Objective")

plt.xlabel("Iterations")
plt.ylabel("Value")






#GENETIC ALGORITHMS
#final_check =  []
#start = time.time()
#for i in [0.20, 0.50, 0.80]:
#    for j in [100,250, 500]:
#        ga_fitness = []
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = genetic_alg(problem, pop_size = j, mutation_prob = i ,max_attempts = 500, max_iters = 500, curve = True, random_state = seed)
#        ga_fitness.append(obj_curve)
#        print(np.mean(np.array(ga_fitness)))
#print(np.mean(np.array(ga_fitness)))

# GENETIC ALGORITHMS
ga_fitness = []
start = time.time()

for seed in random_seed_list:
    best_state, best_fitness, obj_curve = genetic_alg(problem, pop_size = 100, mutation_prob = 0.2 ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
    ga_fitness.append(obj_curve)
print("--- %s seconds ---" % (time.time() - start))

    
plt.plot(np.arange(1,5001), np.mean(np.array(ga_fitness), axis = 0))
plt.title("GENETIC ALGORITHMS: Iterations vs Objective")

plt.xlabel("Iterations")
plt.ylabel("Value")






# MIMIC ALGORITHMS
#mimic_fitness = []
#start = time.time()
#for i in [0.10, 0.30, 0.40]:
#    for j in [100,200, 500]:
#        mimic_fitness = [] 
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = mimic(problem, pop_size = j, keep_pct = i ,max_attempts = 500, max_iters = 500, curve = True, random_state = seed)
#        mimic_fitness.append(obj_curve)
#        print(np.mean(np.array(mimic_fitness)))
#print(np.mean(np.array(mimic_fitness)))



# MIMIC
mimic_fitness = []
start = time.time()

for seed in random_seed_list:
    best_state, best_fitness, obj_curve = mimic(problem,  pop_size = 500, keep_pct = 0.3, max_attempts = 1000, max_iters = 1000, curve = True, random_state = seed, fast_mimic= True)
    mimic_fitness.append(obj_curve)

print("--- %s seconds ---" % (time.time() - start))
    
plt.plot(np.arange(1,1001), np.mean(np.array(mimic_fitness), axis = 0))
plt.title("MIMIC: Iterations vs Objective")

plt.xlabel("Iterations")
plt.ylabel("Value")











#plt.plot(np.arange(1,5001), np.mean(np.array(rhc_fitness), axis = 0), label = "Random Hill Climbing")
#plt.plot(np.arange(1,5001), np.mean(np.array(sa_fitness), axis = 0), label = "Simulated Annealing")
#plt.plot(np.arange(1,501), np.mean(np.array(ga_fitness), axis = 0) ,label = "GA")
#plt.plot(np.arange(1,201), np.mean(np.array(mimic_fitness), axis = 0), label = "MIMIC")
#plt.legend()
#plt.title("4 Peaks: Iterations vs Objective")
#plt.show()
#
#


