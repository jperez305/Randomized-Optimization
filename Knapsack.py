# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:17:39 2021

@author: Joey
"""



import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose.fitness import Knapsack
from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.opt_probs import TSPOpt, DiscreteOpt
import numpy as np
import matplotlib.pyplot as plt
import mlrose 
import random 
random_seed_list = [10 * i for i in range(5)]
import time

ks_len=100

weights=np.random.uniform(10,75,ks_len)
values=np.random.uniform(25,50,ks_len)
  
 
ks_objective = Knapsack(weights, values, max_weight_pct= 0.5)



problem = DiscreteOpt(length = 100, fitness_fn = ks_objective)
problem.get_state()

# OPTIMIZATION STEP

# RHC
#start = time.time()
#for i in [10,50, 100, 200]:
#    sa_fitness = []
#    for seed in random_seed_list:
#        best_state, best_fitness, obj_curve = random_hill_climb(problem,restarts= i,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
#    sa_fitness.append(obj_curve)
#    print(np.mean(np.array(sa_fitness)))
#print(np.mean(np.array(sa_fitness)))

rhc_fitness = []
start = time.time()

for seed in random_seed_list:
    best_state, best_fitness, obj_curve = random_hill_climb(problem, max_attempts = 1000, max_iters = 1000, curve = True, random_state = seed)
    rhc_fitness.append(obj_curve)
print("--- %s seconds ---" % (time.time() - start))

    
plt.plot(np.arange(1,1001), np.mean(np.array(rhc_fitness), axis = 0))
plt.title("Random Hill Climbing: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")
rhc_plot = np.mean(np.array(rhc_fitness))

#
# SIMULATED ALGORITHMS
#start = time.time()
#for i in [10,50, 100, 200, 500, 1000]:
#    for j in [0.1, 0.2, 0.3, 0.5, 0.8]:
#        sa_fitness = []
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = simulated_annealing(problem, schedule= mlrose.GeomDecay(init_temp = i) ,max_attempts = 5000, max_iters = 5000, curve = True, random_state = seed)
#        sa_fitness.append(obj_curve)
#        print(np.mean(np.array(sa_fitness)))
#print(np.mean(np.array(sa_fitness)))

# SIMULATED ANNEALING
sa_fitness = []
start = time.time()

for seed in random_seed_list:
    best_state, best_fitness, obj_curve = simulated_annealing(problem,schedule= mlrose.GeomDecay(init_temp = 200) ,max_attempts = 1000, max_iters = 1000, curve = True, random_state = seed)
    sa_fitness.append(obj_curve)
print("--- %s seconds ---" % (time.time() - start))

    
plt.plot(np.arange(1,1001), np.mean(np.array(sa_fitness), axis = 0))
plt.title("SIMULATED ANNEALING: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")
sa_plot = np.mean(np.array(sa_fitness))


# GENETIC ALGORITHMS
#start = time.time()
#for i in [0.10, 0.30, 0.50, 0.80]:
#    for j in [100,200,300,400, 500]:
#        ga_fitness = []
#        for seed in random_seed_list:
#            best_state, best_fitness, obj_curve = genetic_alg(problem, pop_size = j, mutation_prob = i ,max_attempts = 1000, max_iters = 1000, curve = True, random_state = seed)
#        ga_fitness.append(obj_curve)
#        print(np.mean(np.array(ga_fitness)))
#print(np.mean(np.array(ga_fitness)))

# GENETIC ALGORITHMS
ga_fitness = []
start = time.time()

for seed in random_seed_list:
    best_state, best_fitness, obj_curve = genetic_alg(problem, pop_size = 300, mutation_prob = 0.9 ,max_attempts = 1000, max_iters = 1000, curve = True, random_state = seed)
    ga_fitness.append(obj_curve)

print("--- %s seconds ---" % (time.time() - start))

    
plt.plot(np.arange(1,1001), np.mean(np.array(ga_fitness), axis = 0))
plt.title("GENETIC ALGORITHMS: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")
ga_plot = np.mean(np.array(ga_fitness))



# MIMIC ALGORITHMS
#mimic_fitness = []
#start = time.time()
#for i in [0.10, 0.30, 0.40, 0.8]:
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
    best_state, best_fitness, obj_curve = mimic(problem, max_attempts = 350, keep_pct = 0.90, max_iters = 350, curve = True, random_state = seed, fast_mimic= True)
    mimic_fitness.append(obj_curve)

print("--- %s seconds ---" % (time.time() - start))
    
plt.plot(np.arange(1,351), np.mean(np.array(mimic_fitness), axis = 0))
plt.title("MIMIC: Iterations vs Objective")
plt.xlabel("Iterations")
plt.ylabel("Value")

MIMIC_plot = np.mean(np.array(mimic_fitness))



















#
#
#
#
#
#
#
#import six
#import sys
#sys.modules['sklearn.externals.six'] = six
#from mlrose.fitness import TravellingSales
#from mlrose.algorithms import random_hill_climb
#from mlrose.opt_probs import TSPOpt
#import numpy as np
#import matplotlib.pyplot as plt
#
#import random_optimization
#
#
#dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
#             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
#             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
#             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
#             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
#             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
#             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
#
#
#random_seed_list = [10 * i for i in range(5)] 
#travelling_objective = TravellingSales(distances = dist_list)
#problem = TSPOpt(length = 5, fitness_fn = travelling_objective, maximize = False)
#
#
## OPTIMIZATION STEP
#
## RHC
#rhc_fitness = []
#for seed in random_seed_list:
#    best_state, best_fitness, obj_curve = random_hill_climb(problem, max_attempts = 500, max_iters = 500, curve = True, random_state = seed)
#    rhc_fitness.append(obj_curve)
#
#    
#plt.plot(np.arange(1,501), np.mean(np.array(rhc_fitness), axis = 0))


