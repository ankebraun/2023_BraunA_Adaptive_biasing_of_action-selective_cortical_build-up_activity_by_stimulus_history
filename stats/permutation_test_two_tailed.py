import numpy as np

def perm_test_two_tailed( a, b ):
    nsamples = 10000
    permutation_a_b = np.zeros( [np.size(a), 2] )
    over_a_b = 1
    delta_a_b = np.nanmean(a) - np.nanmean(b) 
    if np.isnan(delta_a_b):
        p_value_a_b = 1
    else:    
        for k in range(nsamples):
            for i in range(0,np.size(a)):
                permutation_a_b[i,:] = np.random.permutation([a[i], b[i]])
            delta_perm_a_b = np.nanmean(permutation_a_b, axis=0)[0] - np.nanmean(
                permutation_a_b, axis=0)[1]
            if abs(delta_perm_a_b) >= abs(delta_a_b):
                over_a_b += 1
        p_value_a_b = float(over_a_b)/(nsamples + 1)
    return p_value_a_b
