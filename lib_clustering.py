from __future__ import division
import numpy as np


# Epanechnikov kernel function.
def K(v):
    return 3/4*max(0,1-v*v)
    
    
# S_j(t) function.
# ---
# t - argument.
# j - power.
# y - data matrix.
# b - bandwidth.
# n - number of data points for each function.
def S(t, j, y, b, n):
    return sum([((t-i/n)**j) * K((i/n-t)/b) for i in range(1, n+1, 1)])
    
    
# w_i(t) function.
# ---
# t - argument.
# i - index.
# y - data matrix.
# b - bandwidth.
# n - number of data points for each function.
def w(t, i, y, b, n):
    return K((i/n - t)/b) * (S(t, 2, y, b, n) - (t - i/n)*S(t, 1, y, b, n))
    
    
# Returns the list of associated hat matrices H(b) for specified array of values of b.
# ---
# b_array - array of bandwidth values.
# y - data matrix.   
# n - number of data points for each function.
def get_associated_hat_matrix(b_array, y, n):
    return [np.array([[w(i/n, j, y, b, n) for j in range(1, n+1, 1)] for i in range(1, n+1, 1)]) for b in b_array]
    
    
# Returns the list of hat(Y) data matrices for specified array of values of b.
# ---
# b - bandwidth.
# y - data matrix.    
# H_matrix - the H(b) associated hat matrix.
# p - number of functions to cluster over.
# n - number of data points for each function.
def get_Y_hat(y, H_matrix_list, p, n):
    return [np.array([[sum([H_matrix_list[l][j][k]*y[i][k] for k in range(n)]) for j in range(n)] for i in range(p)]) for l in range(len(H_matrix_list))]
    
    
# Returns the list of p x n matrices of e_{k,i} values for specified array of values of b.
# ---
# b_array - array of bandwidth values.
# y - data matrix.    
# mu_values_list - list of values mu_k(i/n).
# p - number of functions to cluster over.
# n - number of data points for each function.        
def get_e_vectors(b_array, y, Y_hat_list, p, n):
    return [np.array([[y[k][i]-Y_hat_list[l][k][i] for i in range(n)] for k in range(p)]) for l in range(len(b_array))]
    
    
# Returns the list of covariance matrices Gamma_n for specified array of values of b.
# ---    
# b_array - array of bandwidth values.
# e_vector_list - list of values e_{k,i}.
# p - number of functions to cluster over.
# n - number of data points for each function. 
def get_Gamma_n(b_array, e_vectors_list, p, n):    
    return [np.array([[[e_vectors_list[l][k][i]*e_vectors_list[l][k][j] for i in range(n)] for j in range(n)] for k in range(p)]) for l in range(len(b_array))]
    
    
# Returns the list of inverse Gamma_n matrices for specified array of values of b.
# ---
# b_array - array of bandwidth values.
# Gamma_n_list - the list of groups of p (n x n) Gamma_n matrices.
# p - number of functions to cluster over.
# n - number of data points for each function. 
def get_Gamma_n_inv(b_array, Gamma_n_list, p, n):
    return [np.array([np.linalg.inv(Gamma_n_list[l][k]) for k in range(p)]) for l in range(len(b_array))]
    
    
# Returns the list of the GCV function values for specified array of values of b.
# ---
# b_array - array of bandwidth values.
# H_matrix_list - the H(b) associated hat matrix.
# Gamma_n_inv_list - the list of groups of p (n x n) inverse Gamma_n matrices.
# Y_hat_list - the list of Y_hat matrices.
# p - number of functions to cluster over.
# n - number of data points for each function. 
def get_GCV(b_array, y, H_matrix_list, Gamma_n_inv_list, Y_hat_list, p, n):
    H_matrix_trace_list = [np.trace(H_matrix_list[l]) for l in range(len(b_array))]
    return [sum([1/(p*n)*np.matmul(np.transpose(Y_hat_list[l][k]-y[k]), np.matmul(Gamma_n_inv_list[l][k],Y_hat_list[l][k]-y[k]))/((1-H_matrix_trace_list[l]/n)**2) for k in range(p)]) for l in range(len(b_array))]
    
    
# Returns the list of indices that, in order, are removed from the specified subset of {1, 2, ..., p} set to obtain S_1, S_2 sets.
# ---
# subset - a subset of {1, 2, ..., p).
# y - data matrix.    
# Y_hat - the Y_hat matrix.
# n - number of data points for each function. 
def get_removed_S_indices(subset, y, Y_hat, n):
    indices_list, remaining_list = [], subset
    while len(remaining_list) > 1:
        average_Y_hat = np.array([np.mean([Y_hat[l][i] for l in remaining_list]) for i in range(n)])
        RSS_list = np.array([sum([(y[k][i]-average_Y_hat[i])**2 for i in range(n)]) for k in remaining_list])
        index = remaining_list[np.where(RSS_list==min(RSS_list))[0][0]]
        indices_list.append(index)
        remaining_list.remove(index)
    indices_list.append(remaining_list[0])
    return indices_list
    
    
# Returns the first cluster obtained by minimizing EBIC from the specified subset of {1, 2, ..., p}. 
# ---
# subset - a subset of {1, 2, ..., p).
# b - the bandwidth value.
# gamma - the gamma parameter for the EBIC.
# y - data matrix.   
# Y_hat - the Y_hat matrix.
# p - number of functions to cluster over.
# n - number of data points for each function. 
def get_first_cluster(subset, b, gamma, y, Y_hat, p, n):
    removed_S_indices_list = get_removed_S_indices(subset, y, Y_hat, n)
    S_list, current_indices_list = [], removed_S_indices_list[:]
    for i in removed_S_indices_list:
        S_list.append(current_indices_list[:])
        current_indices_list.remove(i)
    average_Y_hat_S_list = np.array([[np.mean([Y_hat[l][i] for l in S]) for i in range(n)] for S in S_list])
    RSS_S_list = np.array([sum([sum([(y[k][i]-average_Y_hat_S_list[j][i])**2 for i in range(n)]) for k in S]) for j, S in enumerate(S_list)])
    RSS_i_list = np.array([sum([(y[k][i]-average_Y_hat_S_list[j][i])**2 for i in range(n)]) for j, k in enumerate(removed_S_indices_list)])
    EBIC_array = np.array([n*p*np.log10(rss_s + sum(RSS_i_list[(len(removed_S_indices_list)-i):]))+(1/b-1)*(i+1)*((1-gamma)*np.log10(n*b)+gamma*np.sqrt(n*b)) for i, rss_s in enumerate(RSS_S_list)])
    return S_list[np.where(EBIC_array == min(EBIC_array))[0][0]]
    
    
# Returns the partition (list of lists) obtained by minimizing EBIC.
# ---
# b - the bandwidth value.
# gamma - the gamma parameter for the EBIC.
# y - data matrix.   
# Y_hat - the Y_hat matrix.
# p - number of functions to cluster over.
# n - number of data points for each function. 
def get_partition(b, gamma, y, Y_hat, p, n):
    partition, current_indices_list = [], range(p)
    while len(current_indices_list) > 0:
        current_cluster = get_first_cluster(current_indices_list[:], b, gamma, y, Y_hat, p, n)
        partition.append(current_cluster[:])
        current_indices_list = [i for i in current_indices_list if not i in current_cluster]
    return partition
    
    






# Clustering function.
# ---
# y - data matrix.
# bandwidth - bandwidth b_n value on the interval (0,1). Can be either a number or an array. If an array, then the bandwidth will be selected from this array by minimizing the GCV value (WARNING: may take a long time for larger datasets).
# gamma - value for the gamma in the EBIC formula.
# DEBUG - whether to print out the diagnostic information.
def get_clusters(y, bandwidth=0.5, gamma=1, DEBUG=False):
    
    # Calculate the dimensionality of data (p - number of functions to cluster; n - number of data points in each function).
    p = len(y)
    n = len(y[0])
    
    # Set up the function for diagnostic output.
    def Print(string):
        if DEBUG:
            print string
   
    # This part of code will only be run if bandwidth is an array.
    if hasattr(bandwidth, '__len__'):
        b_array = bandwidth
        Print("STEP 1: DETERMINING THE OPTIMAL BANDWIDTH VALUE")
    
        # Calculate the associated hat matrix for hat(Y)_k(b) for all present values of b.
        Print("Calculating the list of the associated hat matrices H(b)...")
        H_matrix_list = get_associated_hat_matrix(b_array, y, n)
        Print("Done.")
        
        # Calculate the hat(Y) data matrix for all present values of b.
        Print("Calculating the list of the hat(Y) values...")
        Y_hat_list = get_Y_hat(y, H_matrix_list, p, n)
        Print("Done.")
        
        # Calculate the e_{k,i} data matrices for all present values of b.
        Print("Calculating the list of the e_{k,i} values...")
        e_vectors_list = get_e_vectors(b_array, y, Y_hat_list, p, n)
        Print("Done.")
        
        # Calculate the Gamma_n covariance matrices for all present values of b.
        Print("Calculating the list of the Gamma_n matrices...")
        Gamma_n_list = get_Gamma_n(b_array, e_vectors_list, p, n)
        Print("Done.")
        
        # Calculate the inverse Gamma_n matrices for all present values of b.
        Print("Calculating the list of inverse Gamma_n matrices...")
        Gamma_n_inv_list = get_Gamma_n_inv(b_array, Gamma_n_list, p, n)
        Print("Done.")
        
        # Calculate the list of GCV values for all present values of b.
        Print("Calculating the list of GCV values...")
        GCV_list = get_GCV(b_array, y, H_matrix_list, Gamma_n_inv_list, Y_hat_list, p, n)
        Print("Done.")
        
        # Get the bandwidth value minimizing the GCV function, as well as its index.
        b, b_index = b_array[np.where(GCV_list==min(GCV_list))[0][0]], np.where(GCV_list==min(GCV_list))[0][0]
        Print("The value of bandwidth minimizing the GCV function is: "+str(b))
        Print("Setting b="+str(b)+".")
        Print("")
        # Fixing the associated Y_hat matrix:
        Y_hat = Y_hat_list[b_index]
        Print("STEP 2: PERFORMING THE CLUSTERING PROCEDURE")
    else:
        b = bandwidth
        Print("STARTING THE CLUSTERING PROCEDURE.")
        Print("Using the bandwidth value b="+str(b)+".")
        # Calculating the associated hat matrix for the specific bandwidth value:
        H_matrix = get_associated_hat_matrix([b], y, n)[0]
        Y_hat = get_Y_hat(y, [H_matrix], p, n)[0]
    Print("Using the gamma value "+str(gamma)+".")
    
    
    # Now, perform the clustering procedure.
    EBIC_minimizing_partition = get_partition(b, gamma, y, Y_hat, p, n)
    Print("Resulting partition: " + str(EBIC_minimizing_partition))
    
    return EBIC_minimizing_partition