from __future__ import division
import numpy as np
import lib_clustering
import lib_simulation
import csv
import time


# ---
# This program runs a large number of simulation instances for all possible combinations of specified parameters and computes the mean and median adjusted Rand Index for each. The adjusted Rand Indexes then are saved to a .csv file.
# ---


# .csv file name:
csv_filename = 'sim_ARI.csv'

# Lists of values for sigma_squared (noise-to-signal level), s (smoothness), gamma (EBIC parameter) and b (bandwidth) to consider:
sigma_squared = [1, 2]
s = [0, 1]
#gamma = [0.25, 0.5, 1]
gamma = [0.5]
#b = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
b = [0.3]

# Lists of numbers of functions p and numbers of equally spaced time points n in each function. Plus the list of fractions f separating different clusters of functions (see the description of get_rademacher_gaussian_simulation_data() function in lib_simulation.py file.
#p = [50, 100]
p = [50]
#n = [500]
n = [100]
f = [0.4, 0.8]

# Number of simulation iterations for each combination of values.
N = 10

# After how many simulation instances to print an update.
reporting_freq = 1






if __name__ == '__main__':

    time_begin = time.time()
    def Print(s):
        print str(int(time.time()-time_begin))+' sec:        ' + s

    savefile = open(csv_filename,'wb')    
    wr_savefile = csv.writer(savefile, quoting=csv.QUOTE_NONE)
    wr_savefile.writerow(['Value'] + ['p'] + ['n'] + ['sigma_squared'] + ['s'] + ['gamma'] + ['b'] + ['Result'])
    N_total_instances = int(len(p)*len(n)*len(sigma_squared)*len(s)*len(gamma)*len(b)*N)
    Print("Total number of simulation instances to run: " +str(N_total_instances))
    count = 0
    for v_p in p:
        for v_n in n:
            for v_sigma_squared in sigma_squared:
                for v_s in s:
                    for v_gamma in gamma:
                        for v_b in b:
                            wr_savefile.writerow([] + [] + [] + [] + [] + [] + [] + [])
                            ARI_array = []
                            for i in range(N):
                                y, true_partition = lib_simulation.get_rademacher_gaussian_simulation_data (v_p, v_n, f, v_sigma_squared, v_s)
                                partition = lib_clustering.get_clusters(y, bandwidth=v_b, gamma=v_gamma)
                                ARI_array.append(lib_simulation.get_adjusted_rand_index (true_partition, partition))
                                count += 1
                                if count % reporting_freq == 0:
                                    Print(str(count) + ' out of ' + str(N_total_instances) + '  (' + str(round(count*100/N_total_instances,2)) + '%)  done.')
                            ARI_array = np.array(ARI_array)
                            wr_savefile.writerow(['Mean ARI'] + [v_p] + [v_n] + [v_sigma_squared] + [v_s] + [v_gamma] + [v_b] + [np.mean(ARI_array)])
                            wr_savefile.writerow(['Median ARI'] + [v_p] + [v_n] + [v_sigma_squared] + [v_s] + [v_gamma] + [v_b] + [np.median(ARI_array)])
    Print ('Simulation finished. The results have been saved in '+csv_filename+'.')