from __future__ import division
import numpy as np
import math
from sklearn import metrics
from scipy.special import legendre


# Calculates the adjusted rand index of the test partition versus the true partition. Each partition is a list of lists of data point indeces.
# ---
# true_partition - true partition of the functions.
# test_partition - the partition to compare to the true one.
def get_adjusted_rand_index (true_partition, test_partition):
    indeces = [l for L in true_partition for l in L]
    if not len(indeces) == len([l for L in test_partition for l in L]) or not set(indeces) == set([l for L in test_partition for l in L]):
        return np.nan
    true_indeces, test_indeces = np.zeros(len(indeces)), np.zeros(len(indeces))
    for i, array in enumerate(true_partition):
        for j, index in enumerate(indeces):
            if index in array:
                true_indeces[j]=i
    for i, array in enumerate(test_partition):
        for j, index in enumerate(indeces):
            if index in array:
                test_indeces[j]=i
    return metrics.adjusted_rand_score(list(true_indeces), list(test_indeces))
    
    
# Generates a simulation dataset as described in Zhang (2013) on page 582. Returns an array of two elements, the first one being the generated dataset, and the second being the true partition.
# ---
# p - Number of functions to generate. Typical values: [50-100].
# n - Number of (equally spaced) time points in each function. Typical values: [100-500].
# split_list - list of fractions separating different clusters of functions. For example, for clusters 1, 2 and 3 to contain 17$, 39% and 44% functions respectively, set split_list=[0.17,0.39].
# sigma_squared - noise-to-signal level. Typical values: [1,2].
# s - level of smoothness. Typical values: [0,1].
def get_rademacher_gaussian_simulation_data (p, n, split_list, sigma_squared, s):
    y, true_partition = [], []
    if len(split_list) == 0:
        true_partition = [range(p)]
    else:
        for i, r in enumerate(split_list):
            if i==0:
                true_partition.append(range(int(math.floor(r*p))))
            else:
                true_partition.append(range(max(true_partition[i-1])+1,int(math.floor(r*p)),1)) 
        true_partition.append(range(max(true_partition[i])+1,p,1))
    for q, P in enumerate(true_partition):
        l_legendre = legendre(s+q) 
        for k in P:
            nu_list, nu = [], np.random.normal(0,1)
            nu_list.append(nu)
            for i in range(1, n, 1):
                nu = (0.5*(i+1)/n-0.2)*nu + np.random.normal(0,1)
                nu_list.append(nu)    
            y.append([l_legendre(2*(i+1)/n-1)+(k+1)-3-5*math.floor((k+1)/5)+2*np.sqrt(sigma_squared)*(((i+1)/n-0.5)**2)*nu_list[i] for i in range(n)])
    y = np.array(y)
    return [y, true_partition]