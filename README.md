# TimeSeriesClustering
Implements a clustering method for high-dimensional sets of trend lines based on trend similarities.
Based on the "Clustering High-Dimensional Time Series Based on Parallelism" by T. Zhang (2013), JASA.




--- PREREQUISITES ---

Python 2.7, Numpy, Math, SciPy, scikit-learn, csv, time




--- INSTALLATION ---

Simply copy the files to the directory from which you want to work with them (for example, your project's directory).




--- USAGE ---

The project contains three user-friendly functions:
  
  
  1. Clustering function.
  
        Takes in a 2-dimensional array of time series. Returns the partition (list of lists of indices) obtained by minimization of the EBIC value.
        
        Example of usage:
        
        ```python
        import numpy as np
        from lib_clustering import get_clusters
        y = np.random.randint(1,11,(50,100))
        print "Partition:", get_clusters(y, bandwidth=0.5, gamma=1, DEBUG=True)
        ```
       
       
  2. Adjusted Rand Index Function.
  
        Takes in two partitions (lists of lists of indices), one of which is the true partition, and calculates the adjusted Rand Index for them.
        
        Example of usage:
        
        ```python
        from lib_simulation import get_adjusted_rand_index
        true_partition = [[1,2,3,4], [5,6,7], [8,9]]
        test_partition = [[1,3,4],[6,7,8], [5,9]]
        print "Adjusted Rand Index:", get_adjusted_rand_index(true_partition, test_partition)
        ```
        
        
  3. Rademacher/Gaussian simulation data generator.
  
        Takes in a number of parameters:
        - number of time series p
        - number of time points in each time series n
        - list of fractions separating the series into groups split_list
        - noise-to-signal ratio sigma_squared
        - smoothness s
        and returns a list of two elements: the array of the generated time series, and the true partition of the series.
        
        Example of usage:
        
        ```python
        from lib_simulation import get_rademacher_gaussian_simulation_data
        p = 50
        n = 100
        split_list = [0.2, 0.6]
        sigma_squared = 1
        s = 1
        y, true_partition = get_rademacher_gaussian_simulation_data (p, n, split_list, sigma_squared, s)
        print "True partition:", true_partition
        print "Generated time series:", y
        ```
        
        
  File run_simulation.py can be run directly: 
  ```bash
  python run_simulation.py
  ```
  It runs a number of simulations with different combinations of parameter values (which can be specified in the file itself), multiple instances of which, and produces a .csv-file containing mean and median adjusted Rand Indices for each combination.
