from libc cimport math
cimport cython
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
import sys

np.import_array()
cdef extern from "numpy/npy_math.h":
    float NPY_INFINITY

from scipy.sparse import dok_matrix


# decorators
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef computeGabriel(
        np.ndarray[np.float32_t, ndim=2] D2, int verbose=0):
    """Compute Gabriel graph from square matrix of squared, pairwise distances.
    Parameters
    ----------
    D2 : array-like, shape (n_samples, n_samples)
        Squared Euclidean distances between all training samples.
    verbose : int
        Verbosity level.
    Returns
    -------
    A : dok_matrix, shape (n_samples, n_samples)
        Adjacency matrix
    """
    cdef unsigned long N = D2.shape[0]
    cdef double dij, m2, eps
    cdef unsigned long xi, xj, xk
    cdef int connect

    eps = 10*np.finfo(D2.dtype).eps

    A = dok_matrix((N, N), dtype=np.uint8)

    if verbose: verbose = max(verbose,100) #use verbose val as step in loggin # of nodes seen 
    
    for xi in range(N):
        if verbose and ((xi + 1) % verbose == 0 or xi + 1 == N):
            sys.stdout.write("%d " % (xi+1))

        
        for xj in range(xi+1,N):
            #check if any point is closer to both xi and xj than D2[xi,xj]
            dij = D2[xi,xj] + eps
            connect = 1
            for xk in range(N):
                if xk == xi or xk == xj: continue
                m2 = D2[xi,xk] + D2[xj,xk]
                if m2 <= dij:
                    connect = 0
                    break
            if connect:
                A[xi,xj] = 1

    A = A.maximum(A.transpose())

    if verbose:
        print("Done.")
    return A

# decorators
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef computeGabriel64(
        np.ndarray[np.float64_t, ndim=2] D2, int verbose=0):
    """Compute Gabriel graph from square matrix of squared, pairwise distances.
    Parameters
    ----------
    D2 : array-like, shape (n_samples, n_samples)
        Squared Euclidean distances between all training samples.
    verbose : int
        Verbosity level.
    Returns
    -------
    A : dok_matrix, shape (n_samples, n_samples)
        Adjacency matrix
    """
    cdef unsigned long N = D2.shape[0]
    cdef double dij, m2, eps
    cdef unsigned long xi, xj, xk
    cdef int connect

    eps = 10*np.finfo(D2.dtype).eps

    A = dok_matrix((N, N), dtype=np.uint8)

    if verbose: verbose = max(verbose,100) #use verbose val as step in loggin # of nodes seen 
    
    for xi in range(N):
        if verbose and ((xi + 1) % verbose == 0 or xi + 1 == N):
            sys.stdout.write("%d " % (xi+1))

        
        for xj in range(xi+1,N):
            #check if any point is closer to both xi and xj than D2[xi,xj]
            dij = D2[xi,xj] + eps
            connect = 1
            for xk in range(N):
                if xk == xi or xk == xj: continue
                m2 = D2[xi,xk] + D2[xj,xk]
                if m2 <= dij:
                    connect = 0
                    break
            if connect:
                A[xi,xj] = 1

    A = A.maximum(A.transpose())

    if verbose:
        print("Done.")
    return A

# decorators
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=1] greedySplitting(
        dict weighted_edges, np.ndarray[np.float32_t, ndim=1] r_FNs, float C=1.,
        int verbose=0):
    """ Greedy algorithm for computing optimal individual kernel scales. 
    (loosely approximates what is achieved by the linear program 
    from the IAN kernel algorithm, but considerably faster)
    Parameters
    ----------
    weighted_edges : dict
        Python dictionary with edge (i,j) as key and distance(i,j) as value
    r_FNs : ndarray, shape (n_samples,)
        A list containing the distance to the furthest neighbor of each data point
    C: float
        The C-connectivity parameter described in the IAN algorithm.
    verbose: int
        Prints outs how many nodes have been processed, in intervals given by max(verbose,100).

    Returns
    -------
    A : ndarray, shape (n_samples,)
        A list containing the individual scales computed by the algorithm.
    """

    cdef unsigned long N = r_FNs.size
    

    cdef np.ndarray[np.float32_t, ndim=1] sigmas = np.zeros(N, dtype=np.float32)
    cdef double rij, sisj, eps
    cdef unsigned long cc, xi, xj
    cdef int xi_exceeds_FN, xj_exceeds_FN

    eps = 10*np.finfo(np.float32).eps
    cdef double FEAS_TOL = 1e-8

    if verbose: verbose = max(verbose,100) #use verbose val as step in loggin # of nodes seen 

    cc = 0
    for (xi,xj),rij in weighted_edges.items():
        cc += 1
        if verbose and (cc % verbose == 0 or cc == N):
                    sys.stdout.write("%d " % cc)

        sisj = sigmas[xi] * sigmas[xj]


        if sigmas[xi] + sigmas[xj] == 0: #neither assigned
            #"split" the distance between the two sigmas
            sigmas[[xi,xj]] = C*rij

        else:
            if sisj > 0: #both_assigned
                if sisj < (C*rij)**2 - eps:#if they dont cover this edge
                    #augment them equally so they cover this edge
                    a = C*rij/math.sqrt(sisj)
                    #assert a >= 1
                    sigmas[xi] = sigmas[xi] * a
                    sigmas[xj] = sigmas[xj] * a
            else:
                if sigmas[xi] == 0:
                    sigmas[xi] = (C*rij)**2/sigmas[xj]
                elif sigmas[xj] == 0:
                    sigmas[xj] = (C*rij)**2/sigmas[xi]

            xi_exceeds_FN = int(sigmas[xi] > r_FNs[xi] + FEAS_TOL)
            xj_exceeds_FN = int(sigmas[xj] > r_FNs[xj] + FEAS_TOL)
            assert not (xi_exceeds_FN and xj_exceeds_FN)
            
            if xi_exceeds_FN:
                sisj = sigmas[xi] * sigmas[xj]
                sigmas[xi] = r_FNs[xi]
                sigmas[xj] = sisj/sigmas[xi]
            elif xj_exceeds_FN:
                sisj = sigmas[xi] * sigmas[xj]
                sigmas[xj] = r_FNs[xj]
                sigmas[xi] = sisj/sigmas[xj]

    if verbose:
        print("Done.")
    return sigmas

# decorators
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] greedySplitting64(
        dict weighted_edges, np.ndarray[np.float64_t, ndim=1] r_FNs, float C,
        int verbose=0):
    """ Greedy algorithm for computing optimal individual kernel scales. 
    (loosely approximates what is achieved by the linear program 
    from the IAN kernel algorithm, but considerably faster)
    Parameters
    ----------
    weighted_edges : dict
        Python dictionary with edge (i,j) as key and distance(i,j) as value
    r_FNs : ndarray, shape (n_samples,)
        A list containing the distance to the furthest neighbor of each data point
    C: float
        The C-connectivity parameter described in the IAN algorithm.
    verbose: int
        Prints outs how many nodes have been processed, in intervals given by max(verbose,100).

    Returns
    -------
    A : ndarray, shape (n_samples,)
        A list containing the individual scales computed by the algorithm.
    """
    cdef unsigned long N = r_FNs.size
    

    cdef np.ndarray[np.float64_t, ndim=1] sigmas = np.zeros(N, dtype=np.float64)
    cdef double rij, sisj, eps
    cdef unsigned long cc, xi, xj
    cdef int xi_exceeds_FN, xj_exceeds_FN

    eps = 10*np.finfo(np.float64).eps
    cdef double FEAS_TOL = 1e-8

    if verbose: verbose = max(verbose,100) #use verbose val as step in loggin # of nodes seen 

    cc = 0
    for (xi,xj),rij in weighted_edges.items():
        cc += 1
        if verbose and (cc % verbose == 0 or cc == N):
                    sys.stdout.write("%d " % cc)

        sisj = sigmas[xi] * sigmas[xj]


        if sigmas[xi] + sigmas[xj] == 0: #neither assigned
            sigmas[[xi,xj]] = C*rij

        else:
            if sisj > 0: #both_assigned
                if sisj < (C*rij)**2 - eps:#if they dont cover this edge
                    #fix them by augmenting them equally so they cover this edge
                    a = C*rij/math.sqrt(sisj)
                    #assert a >= 1
                    sigmas[xi] = sigmas[xi] * a
                    sigmas[xj] = sigmas[xj] * a
            else:
                if sigmas[xi] == 0:
                    sigmas[xi] = (C*rij)**2/sigmas[xj]
                elif sigmas[xj] == 0:
                    sigmas[xj] = (C*rij)**2/sigmas[xi]

            xi_exceeds_FN = int(sigmas[xi] > r_FNs[xi] + FEAS_TOL)
            xj_exceeds_FN = int(sigmas[xj] > r_FNs[xj] + FEAS_TOL)
            assert not (xi_exceeds_FN and xj_exceeds_FN)

            if xi_exceeds_FN:
                sisj = sigmas[xi] * sigmas[xj]
                sigmas[xi] = r_FNs[xi]
                sigmas[xj] = sisj/sigmas[xi]
            elif xj_exceeds_FN:
                sisj = sigmas[xi] * sigmas[xj]
                sigmas[xj] = r_FNs[xj]
                sigmas[xi] = sisj/sigmas[xj]

    if verbose:
        print("Done.")
    return sigmas


cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5

cpdef np.ndarray[np.float32_t, ndim=2] _binary_search_perplexity(
        np.ndarray[np.float32_t, ndim=2] sqdistances,
        float desired_perplexity,
        int verbose):
    """Binary search for sigmas of conditional Gaussians.
    This approximation reduces the computational complexity from O(N^2) to
    O(uN). Code adapted from the scikit-learn package: https://scikit-learn.org/

    Parameters
    ----------
    sqdistances : array-like, shape (n_samples, n_neighbors)
        Distances between training samples and their k nearest neighbors.
        When using the exact method, this is a square (n_samples, n_samples)
        distance matrix. The TSNE default metric is "euclidean" which is
        interpreted as squared euclidean distance.
    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : array, shape (n_samples, n_samples)
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_samples = sqdistances.shape[0]
    cdef long n_neighbors = sqdistances.shape[1]
    cdef int using_neighbors = n_neighbors < n_samples
    # Precisions of conditional Gaussian distributions
    cdef double beta
    cdef double beta_min
    cdef double beta_max
    cdef double beta_sum = 0.0

    # Use log scale
    cdef double desired_entropy = math.log(desired_perplexity)
    cdef double entropy_diff

    cdef double entropy
    cdef double sum_Pi
    cdef double sum_disti_Pi
    cdef long i, j, k, l

    # This array is later used as a 32bit array. It has multiple intermediate
    # floating point additions that benefit from the extra precision
    cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros(
        (n_samples, n_neighbors), dtype=np.float64)

    for i in range(n_samples):
        beta_min = -NPY_INFINITY
        beta_max = NPY_INFINITY
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed just over the nearest neighbors or over all data
            # if we're not using neighbors
            sum_Pi = 0.0

            #1) add kernel-similarity vals from all nbrs
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            #2) normalize simis into probs
            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == NPY_INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -NPY_INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta

        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n_samples):
            print("[t-SNE] Computed conditional probabilities for sample "
                  "%d / %d" % (i + 1, n_samples))

    if verbose:
        print("[t-SNE] Mean sigma: %f"
              % np.mean(math.sqrt(n_samples / beta_sum)))
    return P

cpdef np.ndarray[np.float32_t, ndim=1] get_tsne_sigmas(
        np.ndarray[np.float32_t, ndim=2] sqdistances,
        float desired_perplexity,
        int verbose):
    """Binary search for sigmas of conditional Gaussians.
    This approximation reduces the computational complexity from O(N^2) to
    O(uN). Code adapted from the scikit-learn package: https://scikit-learn.org/
    Parameters
    ----------
    sqdistances : array-like, shape (n_samples, n_neighbors)
        Distances between training samples and their k nearest neighbors.
        When using the exact method, this is a square (n_samples, n_samples)
        distance matrix. The TSNE default metric is "euclidean" which is
        interpreted as squared euclidean distance.
    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : array, shape (n_samples, n_samples)
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_samples = sqdistances.shape[0]
    cdef long n_neighbors = sqdistances.shape[1]
    cdef int using_neighbors = n_neighbors < n_samples
    # Precisions of conditional Gaussian distributions
    cdef double beta
    cdef double beta_min
    cdef double beta_max
    cdef double beta_sum = 0.0

    # Use log scale
    cdef double desired_entropy = math.log(desired_perplexity)
    cdef double entropy_diff

    cdef double entropy
    cdef double sum_Pi
    cdef double sum_disti_Pi
    cdef long i, j, k, l

    # This array is later used as a 32bit array. It has multiple intermediate
    # floating point additions that benefit from the extra precision
    cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros(
        (n_samples, n_neighbors), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] sigmas = np.zeros(
        (n_samples), dtype=np.float64) #array to store sigmas

    for i in range(n_samples):
        beta_min = -NPY_INFINITY
        beta_max = NPY_INFINITY
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed just over the nearest neighbors or over all data
            # if we're not using neighbors
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == NPY_INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -NPY_INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0
        sigmas[i] = math.sqrt(1.0/(2.0*beta))
        beta_sum += beta

        if verbose and ((i + 1) % 1000 == 0 or i + 1 == n_samples):
            print("[t-SNE] Computed conditional probabilities for sample "
                  "%d / %d" % (i + 1, n_samples))

    if verbose:
        print("[t-SNE] Mean sigma: %f"
              % math.sqrt(n_samples / beta_sum))
    return sigmas