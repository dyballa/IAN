import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy as sp
import time
from scipy.spatial.distance import squareform, pdist
from scipy.stats import norm as gauss
from scipy.stats import median_abs_deviation
from collections import deque
from ian.utils import subps, getTri, plotDataGraph
from ian.cutils import computeGabriel, greedySplitting

solver_opts = {'SCS':dict(use_indirect=True,eps=1e-3), 'ECOS':dict(abstol=1e-3,reltol=1e-3),
       'GUROBI':dict(FeasibilityTol=1e-5,OptimalityTol=1e-5)}

def getSparseMultiScaleK(D2, optSigs, sig2scl=2., disc_pts=[], tol=1e-8, degrees=None, 
                            debugPts=[],returnSparse=True,symmetrize=True,verbose=False):

    N = len(optSigs)
    
    disc_pts = set(disc_pts)
    
    try:
        assert np.all(np.flatnonzero(np.isclose(optSigs,0)) == sorted(disc_pts))
    except:
        print('Zero-sigmas assertion in getSparseMultiScaleK failed')
        print(np.flatnonzero(np.isclose(optSigs,0)), sorted(disc_pts))
    
    nonzero_sigs = optSigs.copy()
    nonzero_sigs[np.isclose(nonzero_sigs,0)] = 1
    
    if verbose:
        start = time.time()
    if type(D2) == dict:
        raise NotImplementedError
        eps = 2*np.finfo(D2['distances'][0].dtype).eps
        maxpow = -np.log(eps) 
        compactD2 = 1
        
    else:
        assert D2.shape == (N,N)
        eps = 2*np.finfo(D2.dtype).eps
        maxpow = -np.log(eps) 
        compactD2 = 0

        diagSigs = 1./nonzero_sigs
        
        if sp.sparse.issparse(D2):
            D2 = D2.tocsr()
            diagSigs = sp.sparse.diags(diagSigs)
            power = diagSigs @ D2 @ diagSigs
            power_data = power.data / sig2scl
            res = np.zeros_like(power_data)
            np.exp(-power_data, where=power_data < maxpow, out=res)#avoid underflow warnings
            res[res < tol] = 0
            power.data = res
            K = sp.sparse.csr_matrix((power.data, power.indices, power.indptr),
                             shape=power.shape) + sp.sparse.diags(np.ones(N))
        else:
            power = diagSigs[None,:] * D2 * diagSigs[:,None]
            power /= sig2scl
            K = np.zeros_like(D2)
            np.exp(-power, where=power < maxpow, out=K)#avoid underflow warnings
            K[K < tol] = 0
            if returnSparse:
                K = sp.sparse.csr_matrix(K)

    if symmetrize:
        if returnSparse:
            K = K.maximum(K.transpose())
        else:
            K = np.maximum(K,K.T)
    if verbose:
        print(f'Computing msK done. ({time.time()-start:.2f}s)')   
    return K

def computeMutualAdjacencyMatrix(Adj, nbr_indices, D2, mutual_method):
    # Do not symmetrize the initial k-NN graph (during Gabriel computation, above),
    # except for nearest neighbors. This possibly improves the approximate graph,
    # according to the paper:
    # This may help the Gabriel graph since with partial information about distances
    # it is possible to connect points that would violate the Gabriel rule.
    # The authors suggest adding edges from the minimum-spanning tree to prevent it
    # from being disconnected. Here we implement adaptations of two methods, 
    # `MST-all` and `MST-min`, that are described in the paper.
    
    eps = 10*np.finfo(D2.dtype).eps
    def connectGabriel(xi,xj):
        #check Gabriel rule
        connect = 1
        dij = D2[xi,xj]
        assert dij
        #for ech nearest pt of xi..
        for nni in set(nbr_indices[xi]).intersection(nbr_indices[xj]):
            #see if it's inside the ball B_ij
            d2i, d2j = D2[xi,nni], D2[xj,nni]
            assert d2i*d2j
            if d2i + d2j <= dij + eps:
                connect = 0
        return connect

    #asymmetric k-NN graph
    N, max_nbrhood_size = nbr_indices.shape
    asymkNN = sp.sparse.csr_matrix( (np.ones(N * max_nbrhood_size, dtype='uint8'), (np.repeat(np.arange(N),max_nbrhood_size),nbr_indices.ravel()) ), shape=(N,N))
    #remove from Adj any asymmetric connections 
    mutualAdj = Adj.multiply(asymkNN.minimum(asymkNN.transpose())).tolil()

    #ensure symmetry
    assert mutualAdj.maximum(mutualAdj.transpose()).nnz == mutualAdj.nnz

    #connect NNs
    for xi,nni in enumerate(nbr_indices[:,0]):
        mutualAdj[xi,nni] = 1
        mutualAdj[nni,xi] = 1

    nCCs, CCis = sp.sparse.csgraph.connected_components(mutualAdj)
    print('nCCs',nCCs)
    if nCCs > 1:#TODO: ignore CCs of very small size?

        mst = sp.sparse.csgraph.minimum_spanning_tree(D2)
        nCCs_mst = sp.sparse.csgraph.connected_components(mst)[0]
        print('nCCs_mst',nCCs_mst)
        if nCCs_mst > 1:
            print('***WARNING: Initial # of minimum-spanning tree connected components > 1.')
            print('This indicates you should increase `max_nbrhood_size` or check for outliers.')
            if input("Proceed anyway? Y or N").lower() == 'n':
                raise ValueError('Stopping due to disconnected MST.')
        
        if mutual_method == 'MST-all':

            # compute symmetrized MST
            rows,cols = mst.nonzero()
            mst1s = sp.sparse.csc_matrix((np.ones(len(rows),dtype='uint8'),(rows,cols)),shape=mst.shape)
            mst1s = mst1s.maximum(mst1s.transpose())

            # add all edges from sym MST to mutualAdj
            mutualAdj_ = mutualAdj.maximum(mst1s)

            # ensure that new edges are Gabriel-neighbors
            new_edges = mutualAdj_ - mutualAdj
            rows,cols = mst.nonzero()
            for xi,xj in zip(rows,cols):
                if not connectGabriel(xi,xj):
                    mutualAdj_[xi,xj] = 0
            mutualAdj_.eliminate_zeros()
            mutualAdj = mutualAdj_


        elif mutual_method == 'MST-min':
            # add the min # of edges sufficient to create a single CC,
            # connecting the CCs using the shortest edges possible

            rows,cols,ws = sp.sparse.find(mst)
            idxs = ws.argsort()
            rows = rows[idxs]
            cols = cols[idxs]
            
            connectedCCs = set()
            # traverse each edge in the MST in order of increasing length
            for xi,xj in zip(rows,cols):
                #if xi and xj are in separate CCs ...
                if CCis[xi] != CCis[xj]:
                    cci, ccj = CCis[[xi,xj]]
                    #... and either of their CCs is not yet connected ...
                    if cci in connectedCCs and ccj in connectedCCs:
                        continue
                    #  and they can be Gabriel-neighbors ...
                    if connectGabriel(xi,xj):
                        #... connect them!
                        connectedCCs.union([cci,ccj])
                        mutualAdj[xi,xj] = 1
                        mutualAdj[xj,xi] = 1
                        print('connected',cci,ccj,'nCCs:',sp.sparse.csgraph.connected_components(mutualAdj)[0])

                        if len(connectedCCs) == nCCs-1:
                            #all connected
                            assert sp.sparse.csgraph.connected_components(mutualAdj)[0] == nCCs_mst
                            break
        else:
            raise ValueError("`mutual_method` not recognized. Must be either 'MST-all' or 'MST-min'.")
    
    #ensure symmetry
    assert mutualAdj.maximum(mutualAdj.transpose()).nnz == mutualAdj.nnz

    return mutualAdj

def computeSparseGabriel(sparseD2, verbose=0):
    """Compute approximate Gabriel graph from a sparse (incomplete) square matrix of squared distances.
    Parameters
    ----------
   sparseD2 : csr sparse array, shape (n_samples, n_samples)
        Squared Euclidean distances between all training samples in csr format
    verbose : int
        Verbosity level.
    Returns
    -------
    A : Adjacencies for each point xi in the format dict-of-dicts
    """
    
    assert sp.sparse.issparse(sparseD2)

    N = sparseD2.shape[0]


    eps = 10*np.finfo(sparseD2.dtype).eps

    A = {xi:{} for xi in range(N)}

    if verbose: verbose = max(verbose,100) #use verbose val as step in logging # of nodes seen 

    for xi in range(N):
        if verbose and ((xi + 1) % verbose == 0 or xi + 1 == N):
            print("%d " % (xi+1), end='')

        xjs, dijs = sparseD2[xi].indices, sparseD2[xi].data

        for xj,dij in zip(xjs, dijs):
            if xj <= xi: continue #already checked
                
            #check if any point is closer to both xi and xj than D2[xi,xj]
            connect = 1
            for xk in set(xjs).intersection(sparseD2[xj].indices):
                assert xk != xi and xk != xj
                m2 = sparseD2[xi,xk] + sparseD2[xj,xk]
                if m2 <= dij + eps:
                    connect = 0
                    break
            if connect:
                A[xi][xj] = 1
                A[xj][xi] = 1

    if verbose:
        print("Done.")
    return A

def getVolumeRatios(degrees,D2,sigmas,wG=None,debugPts=[],verbose=False,sig2scl=2.,
                            disc_pts=[]):
    N = len(sigmas)

    assert len(degrees) == N

    if type(D2) == dict:
        assert 'distances' in D2 and 'indices' in D2
        assert len(D2['distances']) == N
        eps = 2*np.finfo(D2['distances'][0].dtype).eps
        compactD2 = 1
    else:
        assert D2.shape == (N,N)
        eps = 2*np.finfo(D2.dtype).eps
        compactD2 = 0

    
    maxpow = -np.log(eps)

    ratios = np.zeros(N)
    sqrtPi_2 = np.sqrt(sig2scl*np.pi)/2
    sampGvols = np.zeros(N)
    degs = np.zeros(N)


    for xi,(d_i,sig) in enumerate(zip(degrees,sigmas)):
        if d_i == 0:#skip disconnected points
            if xi in debugPts:
                print(f"  x{xi} d_i {d_i}")
            continue

        if wG is None:
            
            if compactD2:
                d2s = D2['distances'][xi]
                assert np.isclose(d2s, 0).sum() == 0
            elif sp.sparse.issparse(D2):
                d2s = D2[xi].data
                assert np.isclose(d2s, 0).sum() == 0
            else:
                d2s = D2[xi]

            power = d2s/(sig2scl*sig**2)
            kvals = np.zeros(power.size,dtype=d2s.dtype)
            np.exp(-power, where=power <= maxpow, out=kvals) #avoid underflow warnings
            wij_sum = kvals.sum() + compactD2 #add self weight if compact
        else:
            wij_sum = wG[xi].sum()

        d_i_ = max(2,d_i) #at least 1-D
        dim_ = np.log2(d_i_)

        #final_dims[xi] = dim_
        ratio = wij_sum/d_i_ 
        
        corr = (sqrtPi_2)**dim_

        ratios[xi] = ratio/corr
        if xi in debugPts:
            print(f"  x{xi} d_i {d_i}, (sig={sig:.2f}): delta_i={wij_sum:.2f}/{d_i_}={ratio:.2f} / correction=(sqrt({sig2scl:.1f}pi)/2)^{dim_:.2f} = {corr:.2f} --> delta_i' = {ratios[xi]:.2f}")
    return np.array(ratios)

def getMuStdev(vals,stdev_method):
    n = len(vals)
    if stdev_method == 'mean':
        return np.mean(vals)

    elif stdev_method == 'median':
        return np.median(vals)

    elif stdev_method == 'MAD':
        mu = np.median(vals)
        stdev = median_abs_deviation(vals,scale=1)

    elif stdev_method == 'normalMAD':
        mu = np.median(vals)
        stdev = median_abs_deviation(vals,scale='normal')

    elif stdev_method == 'q1q3':
        q1,q2,q3 = np.percentile(vals,[25,50,75])
        mu = q2
        stdev = q3 - q2 + 1.5*(q3-q1)
        assert nstds == 1

    elif stdev_method == 'symIQR':
        q1,q2 = np.percentile(vals,[25,50])
        q3 = q2 + q2-q1
        mu = q2
        stdev = q3 - q2 + 1.5*(q3-q1)
        assert nstds == 1

    elif stdev_method == 'C2':
        a = min(vals)
        q1,m,q3 = np.percentile(vals,[25,50,75])
        #q3 = m + (m - q1) #symm q3
        b = q3 + (q1 - a) #symm max val

        mu = 0.125*(a + 2*q1 + 2*m + 2*q3 + b)
        stdev = (b-a)/(4*gauss.ppf((n-.375)/(n+.25))) + \
            (q3-q1)/(4*gauss.ppf((.75*n-.125)/(n+.25)))

    elif stdev_method == 'C3':
        q1,m,q3 = np.percentile(vals,[25,50,75])
        #q3 = m + (m - q1) #symm q3
        mu = 0.333*(q1 + m + q3)
        stdev = (q3-q1)/(2*gauss.ppf((.75*n-.125)/(n+.25)))

    elif stdev_method == 'symmC3':
        q1,m = np.percentile(vals,[25,50])
        q3 = m + (m - q1) #symm q3
        mu = 0.333*(q1 + m + q3)
        stdev = (q3-q1)/(2*gauss.ppf((.75*n-.125)/(n+.25)))

    elif stdev_method in ['std','stdev']:
        mu,stdev =  np.mean(vals),np.std(vals)

    else:
        raise ValueError('Method for stdev_method not recognized.')

    return mu,stdev


def getSigmasFromConstraints(u,cinfo,objfoo,solver=None,verbose=False,
                    optVerbose=False):

    '''Recompute sigmas making use of pre-computed constraints'''

    if verbose:
        print('Building constraints...',end=' ')
        start = time.time()

    add_us = cinfo['us'] * u[cinfo['idxsJ']] + cinfo['u_1s'] * cinfo['u_tilde'][cinfo['idxsI']]
    u_slopes = add_us * cinfo['slopes']
    intercepts = (cinfo['b1'] * add_us) + cinfo['b2']

    n = u.size
    x = cp.Variable(n)
    
    constraints = [ cp.multiply(u_slopes,x[cinfo['idxsI']]) - x[cinfo['idxsJ']] <= intercepts ]
    if solver == 'GUROBI-bounds':
        lb = np.zeros(n)
        ub = u
    else:
        A_ = sp.sparse.vstack([sp.sparse.identity(n),-sp.sparse.identity(n)])
        b_ = sp.sparse.csc_matrix((u,(range(n),np.zeros(n))),shape=(2*n,1))

        constraints.append( A_ @ x[:,None] <= b_  )

    if objfoo in ['wsum','non']:
        assert sum_weights is not None
        c = sum_weights
    elif objfoo == 'sum':
        c = np.ones(n)
    else:
        raise ValueError('objective function not recognized')

    objective = c @ x
    prob = cp.Problem(cp.Minimize(objective), constraints)
    assert prob.is_dcp(dpp=True)
    for cons in constraints:
        assert cons.is_dcp(dpp=True)

    if verbose:
        print(f'Done. ({time.time()-start:.2f} s)')
        start = time.time()

    result = prob.solve(solver=solver,
        verbose=bool(optVerbose),**solver_opts.get(solver,{}))
    if verbose:
        print('Solved sigmas. Obj =',result,f'({time.time()-start:.2f} s)')
    return x.value

def computeThreshold(vals, nstds, maxthresh=np.inf, minval=0, plot=True, ax=None, c='b', label=None,
    stdev_method='std', plotQuartiles=False, title='Volume ratio distribution', showThresh=True):

    minthresh = 2.75 #prevents a pathological threshold for non-generic datasets with stdev ~= 0
    # and, furthermore, empirically prevents most 1-D datasets from being inadvertently disconnected.
    #  
    # setting `maxthresh` prevents an artifically high threshold when distribution is very heavy-tailed
    # (empirically, a value of ~5 is sufficient to guarantee connectedness even for 1-D datasets
    # that are sampled uniformly/normally at random, provided median ~= 1. this value drops to ~3 
    # for multi-scale ratios.)

    # 1-D datasets are used here due to the fact that they are the most extreme case in terms of
    # the effect of non-uniformity of sampling on the distributino of distances to nearest neighbors.


    mu,stdev = getMuStdev(vals,stdev_method)
    thresh = max(minthresh, mu+nstds*stdev) #for non-generic datasets with stdev too close to 0
    if np.all(vals <= thresh) and maxthresh < thresh:
        thresh = maxthresh

    if np.isclose(stdev,0): #for non-generic datasets
        edge_xs = [mu-1e-2, mu+1e-2] #set width of single bar in histogram
        bin_edges = edge_xs

    else:
        try: #infer good number of bins
            bin_edges = np.histogram_bin_edges(vals,'fd')
        except: #use 50 as default
            bin_edges = np.linspace(min(vals),max(vals),50)

    if plot:
        if ax is None: 
            ax = plt.subplot(111)
            
        counts,_,_ = ax.hist(vals,bin_edges,lw=1.,color=c,histtype='step',label=label) 
        ax.set_xlabel(f"volume ratio, $\delta_i'$")
        ax.set_ylabel(f"counts")

        if np.isclose(stdev,0):
            ax.set_xlim(mu-.5, mu+.5) #make the single bar look thin


        if showThresh:
            ax.plot([thresh,thresh],[0,counts.max()], c='xkcd:red', ls='--', lw=1, label='threshold')
            ax.legend(loc="upper right")

        if plotQuartiles:
            for qi,q in enumerate([25,50,75]):
                qi = np.percentile(vals,q)
                ax.plot([qi,qi],[0,counts.max()], c='gray', ls=':', lw=1.5 if q == 50 else 1, 
                        label='quartiles' if q == 50 else None)
            ax.legend(loc="upper right")

        ax.set_title(title)

    else:
        counts,_ = np.histogram(vals,bin_edges)
    
    return mu,stdev,thresh,counts

#TODO: convert to class so we have an object that we can run each iteration separately on
#then can easily revert to a previous state. i.e. save initial connectivity then
#apply history of prunings
def IAN(method, X, Adj=None, max_nbrhood_size=None, Xplot=None, plot_interval=1,
       optC=None, obj='sum', stdev_method='C3', metric='euclidean',
       maxPruneAtaTime=.1, nstds=4.5, maxIter=np.inf, mutual_method=None,
       verbose=1, interactive=True,
       pre_deleted_nodes=None, plot_final_stats=True, saveImgs={}, solver=None,
       return_processed_inputs=False, tune_wG_method='median',
       allowMSconvergence=False, plotQuartiles=True, return_stats=False):
    """Compute IAN kernel from a data matrix or pairwise distances.
    Parameters
    ----------
    method: str
        'exact': compute all pairwise distances from the data matrix X
        'exact-precomputed': assume X is a matrix of distances
        'exact-precomputed-sq': assume X is a matrix of squared distances
        'approximate': compute distances to a restricted number of neighbors (k-NN graph)
            set by `max_nbrhood_size`
        'approximate-precomputed': assume X is a tuple (kdists, knbr_indices) as retuned by KDTree
    X : ndarray, shape (n_samples, n_attributes) or (n_samples, n_samples)
        X is a data matrix (with points as rows) or a pairwise distance matrix
    max_nbrhood_size: int
        sets the k-NN graph nbrhood size when using the 'approximate' method
    Xplot: ndarray, shape (n_samples, 2)
        Provides points positions in 2-D to be plotted for debugging
    plot_interval: int
        Show current data graph every `plot_interval` iterations.
    optC: float
        Set a fixed value for the hyperparameter C (not recommended, used only for debugging)
        Leaving the default None will allow IAN to auto-tune it.
    obj str
        Objective function to use. 'sum' uses the convex linear program; 'greedy' uses a
        faster algorithm, but produces suboptimal results.
    stdev_method str
        Method for computing a robust dispersion statistic of volume ratios. This is used
        to automatically set the pruning threshold at every iteration.
    metric str
        Metric used for computing distances when method is not 'precomputed'
    maxPruneAtaTime float or int
        If int >= 1, sets the maximum number of edges that can be pruned per iteration.
        If float < 1, sets the maximum percentage of nodes above threshold that can have one
        of its edges pruned. For best results, use this value as low as possible. Larger values
        make the algorithm converge fastar but also makes the results more imprecise.
    nstds float
        Number of standard deviations above the mean to set the pruning threshold.
    maxIter int
        Sets a maximum number of iterations, after which IAN should stop. If not yet converged 
        abd `interactive` is True, user will be prompted before terminating.
    mutual_method str or None
        If using an 'approximate' method, whether to start with a mutual k-NN graph and 
        add edges from its minimum spanning tree (see docs). Can be set to 'MST-min' or 'MST-all'.
    verbose : int
        Verbosity level.
    interactive : bool
        Whether to confirm with user if should really stop after reaching maxIters or continue.
    pre_deleted_nodes list
        List of points to be deleted from dataset.
    plot_final_stats bool
        Whether to display the final volume ratios after convergence.
    saveImgs dict
        Optional arguments for saving the plots as figures.
    solver str
        Solver to use with cvxpy optimization package.
    return_processed_inputs bool
        Return initial adjacency matrix (Gabriel graph) and distance matrix, without computing IAN.
    tune_wG_method str
        Method for tuning the weighted graph after convergence. Can be 'median' or 'mean'.
    allowMSconvergence bool
        Whether to check at every iteration if the current weighted graph has no outliers, in which
        case the algorithm can converge earlier (but unweighted graph may not be optimal).
    plotQuartiles bool
        Whether to plot the volume ratio quartiles over the histogram.
    return_stats bool
        If True, also return pruning_history, sigma_history, G_stats, wG_stats

    Returns
    -------
    G : Final unweighted graph
    wG: Final weighted graph
    optSigs: optimal individual scales used for weighted graph
    disc_pts: list of points that may have been disconnected (along with their last nbr and scale)

    """

    minimalVerbose = verbose > 0
    extraVerbose = verbose > 1
    excessiveVerbose = verbose > 2

    sig2scl = 1
    optverbose = False
    debugStatsPts = []
    
    def check_precomputed_Adj_format(Adj):
        assert np.all(Adj.diagonal() == 0) #no loops
        if not sp.sparse.issparse(Adj):
            return np.asarray(Adj) #convert from matrix to ndarray
        if minimalVerbose: print('Successfully loaded precomputed adjacency matrix.')
        return Adj

    def check_precondition_distance_matrices(D1, D2=None):
        assert D1.min() >= 0, 'all distances nonnegative'
        
        if D1.shape[1] == D1.shape[0]:
            flat_d1s = squareform(D1)
            min_d1 = flat_d1s.min()

        else:#approximate method
            DRATIO_ACC = 1e-2
            approx_ratios = D1[:,-1]/D1[:,0].astype('d') #ratio: farthest approx pt / nearest nbr
            # min ratio yielding small enough kernel vals (assuming sigma ~= d_NN)
            # to ensure reasonably accurate estimation of delta statistic
            min_safe_ratio = np.sqrt(1-2*np.log(DRATIO_ACC))
            pct_innacurate = 100*np.sum(approx_ratios < min_safe_ratio)/D1.shape[0]
            if extraVerbose: print('%.1f%% of the points are likely to have innacurate kernel estimation. Increase max_nbrhood_size value to improve.' % pct_innacurate)

            min_d1 = D1[:,0].min()

        if np.isclose(min_d1,0):
            raise ValueError("Nearly identical points found (np.isclose(distance,0) == True). Please remove duplicates before running the algorithm.")

        #scale distances to avoid numerical issues (min dist -> 1)
        scl = 1./min_d1
        D1 *= scl
        if D2 is not None:
            D2 *= scl**2
            return D1,D2,scl
        return D1,scl

    def process_input(X,Adj,method,metric,obj,max_nbrhood_size):

        assert obj in ['greedy','sum','wsum','non']

        if method in ['exact', 'exact-precomputed', 'exact-precomputed-sq']:
            N = X.shape[0]
            if method == 'exact':
                D1 = squareform(pdist(X,metric))
                D2 = np.square(D1)

            elif 'precomputed' in method:
                assert X.shape[1] == N #square matrix
                if method == 'exact-precomputed-sq':
                    D2 = X.copy()
                    D1 = np.sqrt(D2)
                else:
                    D1 = X.copy()
                    D2 = np.square(D1)

            D1, D2, scl = check_precondition_distance_matrices(D1,D2)

            if Adj is None:
                Adj = computeGabriel(D2.astype('float32'),verbose=max(500,N//10) if extraVerbose else 0)
                Adj = Adj.maximum(Adj.transpose())
            else:
                assert Adj.shape[0] == D1.shape[0]
                Adj = check_precomputed_Adj_format(Adj)


        elif method in ['approximate','approximate-precomputed']:
            if method == 'approximate':
                N = X.shape[0]
                assert max_nbrhood_size
                if max_nbrhood_size >= N-1:
                    raise ValueError(
                    'max_nbrhood_size chosen includes all points. Please switch to exact method.'
                    )

                D1k,nbr_indices = getD1k(X, max_nbrhood_size, 1)


            elif method == 'approximate-precomputed':
                D1k, nbr_indices = X[0].copy(), X[1].copy()
                N = D1k.shape[0]

                if np.all(D1k[:,0] == 0):
                    #remove self-distances
                    D1k = D1k[:,1:]
                    nbr_indices = nbr_indices[:,1:]
                assert not np.any(D1k[:,0] == 0)
                
                max_nbrhood_size = D1k.shape[1]
                if extraVerbose: print('Found max_nbrhood_size=%d from precomputed data.' % max_nbrhood_size)



            D1k, scl = check_precondition_distance_matrices(D1k)
            #get sparse, N-by-N matrix D1 from D1k,indices
            D1 = sp.sparse.csr_matrix( (D1k.ravel(), (np.repeat(np.arange(N),max_nbrhood_size),nbr_indices.ravel()) ), shape=(N,N))
            #symmetrize
            D1 = D1.maximum(D1.transpose())
            #eps = np.finfo(D1.dtype).eps #save this for later

            D2 = D1.copy()
            D2.data = D2.data**2  
            if excessiveVerbose: print('done D1,D2')

          

            if Adj is None:
                #compute Gabriel from sparse D2
                Adj = computeSparseGabriel(D2,verbose=max(500,N//10) if extraVerbose else 0)
                #convert from dict-of-dicts to sparse matrix
                rows,cols = zip(*[(xi,xj) for xi, d in Adj.items() for xj in d])
                Adj = sp.sparse.csc_matrix((np.ones(len(rows),dtype='uint8'),(rows,cols)),shape=(N,N))

                if mutual_method is not None:
                    Adj = computeMutualAdjacencyMatrix(Adj, nbr_indices, D2, mutual_method)                
            else:
                assert Adj.shape[0] == N
                Adj = check_precomputed_Adj_format(Adj)

            if excessiveVerbose: print('done Adj')


        else:
            raise ValueError('Method not recognized.')

        return Adj, D1, D2, scl


    def get_connectivity_data(Adj, D1, obj, estimateC):

        N = Adj.shape[0]

        if sp.sparse.issparse(Adj):
            D1Adj = Adj.multiply(D1).tocsr()
            D1Adj.eliminate_zeros()
        elif sp.sparse.issparse(D1):
            D1Adj = D1.multiply(Adj).tocsr()
            D1Adj.eliminate_zeros()            
        else:
            D1Adj = sp.sparse.csr_matrix(np.multiply(Adj, D1))
        if excessiveVerbose: print('done D1Adj')

        
        xis,xjs,d1s = sp.sparse.find(sp.sparse.triu(D1Adj,k=1))
        wA = {(xis[i],xjs[i]):d1s[i] for i in range(len(xis))}
        if excessiveVerbose: print('done wA')
        

        sorted_nbrs = {}
        degrees = np.empty(N,dtype=int)
        FN_D1s = np.empty(N)
        if estimateC:
            med_nbr_ds = np.empty(N)
        else:
            med_nbr_ds = None

        start = time.time()
        all_is,all_js,all_d1s = sp.sparse.find(D1Adj)
        idxs = all_is.argsort()
        all_is = all_is[idxs]
        all_js = all_js[idxs]
        all_d1s = all_d1s[idxs]
        
        i = 0
        while i < len(all_is):
            xi = all_is[i]

            nbr_d1_js = []
            while i < len(all_is) and all_is[i] == xi:
                xj = all_js[i]
                d1 = all_d1s[i]
                nbr_d1_js.append( (d1,xj) )
                i += 1
            
            deg = len(nbr_d1_js)
            degrees[xi] = deg
            nbr_d1s, nbr_is = zip(*sorted(nbr_d1_js, reverse=True))
            sorted_nbrs[xi] = deque(nbr_is)
            FN_D1s[xi] = nbr_d1s[0]
            
            if estimateC:
                mid = min(deg-1,max(1,int(np.ceil(deg/2.))-1))
                med_d = nbr_d1s[deg-mid-1]
                med_nbr_ds[xi] = med_d
        if excessiveVerbose: print('done computing degrees and FN_D1s', time.time()-start)


        sum_weights = None
        non_nnd1s = None
        if obj in ['wsum','non']:
            sum_weights = np.ones(N)
            non_nnd1s = np.empty(N,dtype=D1.dtype)

            if sp.sparse.issparse(D1):
                all_is, all_js, all_d1s = sp.sparse.find(D1)
                idxs = all_is.argsort()
                all_is = all_is[idxs]
                all_js = all_js[idxs]
                all_d1s = all_d1s[idxs]
            else:
                all_is, all_js = D1.nonzero()

            i = 0
            while i < len(all_is):
                xi = all_is[i]
                xi_nbrs = set(sorted_nbrs[xi])
                non_nbr_d1s = []

                while i < len(all_is) and all_is[i] == xi:
                    xj = all_js[i]
                    if xj not in xi_nbrs:
                        if sp.sparse.issparse(D1):
                            jd1 = all_d1s[i]
                        else:
                            jd1 = D1[xi,xj]
                        non_nbr_d1s.append(jd1)
                    i += 1
                
                if non_nbr_d1s:
                    non_nbr_min = min(non_nbr_d1s)
                    non_nnd1s[xi] = non_nbr_min
                    w = non_nbr_min
                    #if obj =='wsum':
                    w /= FN_D1s[xi]
                    sum_weights[xi] = 1./w
                else:
                    #no non-neighbors
                    sum_weights[xi] = 1

            print('done non-nbr weights')

        return wA, sorted_nbrs, degrees, FN_D1s, sum_weights, non_nnd1s, med_nbr_ds

    def prune_edges(to_be_pruned, wA, sorted_nbrs, degrees, FN_D1s, non_nnd1s, sum_weights, sorted_stats,
                    cinfo):

        one_edge_per_node = True
    
        pruned_edge_tups = []
        nodes_seen = set()
        
        if extraVerbose:
            print(f'pruning: from ',end='')

        for xi in to_be_pruned:
            assert len(sorted_nbrs[xi]) >= 1

        new_disc_pts = []
        for xi,stati in zip(to_be_pruned,sorted_stats):
            if one_edge_per_node and xi in nodes_seen:
                continue
            
            assert len(sorted_nbrs[xi]) > 0 #node has not already been disconnected
                
            #get farthest nbr
            fnbr = sorted_nbrs[xi][0]#FNs[xi]

            if one_edge_per_node and fnbr in nodes_seen:
                continue

            edge_tup = tuple(sorted([xi,fnbr]))
            assert edge_tup in wA
            #assert edge_tup in edges_idx_dict

            #prnue edge: update data structures

            d1fnbr = wA[edge_tup]


            del wA[edge_tup]
            #del edges_idx_dict[edge_tup]

            #remove edge from constraint matrix
            #need to update all indices > e_is!
            #1) keep an array with all edge row indices: e_idxs = np.arange(len(slopes))
            #2) edge_is points to this array instead of directly to slopes
            #3) update e_idxs: e_idxs[e_idxs > e_is[1]] -= 2
            #4) use indices from e_idxs[e_is]
            if cinfo is not None:

                e_is = cinfo['e_idxs'][cinfo['edge_is'][edge_tup]]
                del cinfo['edge_is'][edge_tup]

                cinfo['idxsI'] = np.delete(cinfo['idxsI'],e_is)
                cinfo['idxsJ'] = np.delete(cinfo['idxsJ'],e_is)
                cinfo['slopes'] = np.delete(cinfo['slopes'],e_is)
                cinfo['b1'] = np.delete(cinfo['b1'],e_is)
                cinfo['b2'] = np.delete(cinfo['b2'],e_is)
                cinfo['us'] = np.delete(cinfo['us'],e_is)
                cinfo['u_1s'] = np.delete(cinfo['u_1s'],e_is)

                cinfo['e_idxs'][cinfo['e_idxs'] > e_is[1]] -= 2


            #assert sorted_nbrs[xi].pop(0) == fnbr
            assert sorted_nbrs[xi].popleft() == fnbr
            if xi in debugStatsPts:
                print('pruned',fnbr,'from',xi)

            #xii = sorted_nbrs[fnbr].index(xi)
            #assert sorted_nbrs[fnbr].pop(xii) == xi
            sorted_nbrs[fnbr].remove(xi)

            if xi in debugStatsPts:
                print('pruned',xi,'from',fnbr)

            #update degree and FN_D1 for affected nodes
            for pti,pti_nbr in [(xi,fnbr),(fnbr,xi)]:

                degrees[pti] -= 1


                if degrees[pti] > 0:
                    
                    new_fnbr = sorted_nbrs[pti][0] 
                    FN_D1s[pti] = D1[pti,new_fnbr]

                    if sum_weights is not None: #obj == 'wsum'
                        if d1fnbr < non_nnd1s[pti]:
                            #non_nnis[pti] = fnbr
                            non_nnd1s[pti] = d1fnbr #update dist to nearest non-nbr
                            w = d1fnbr#/FN_D1s[pti]
                            if obj =='wsum':
                                w /= FN_D1s[pti]
                            if pti in debugStatsPts:
                                print('new weight',pti,w)
                            sum_weights[pti] = 1/w

                else:#disconnected
                    FN_D1s[pti] = 0
                    new_disc_pts.append((pti,pti_nbr))
                    if sum_weights is not None:
                        sum_weights[pti] = 1 #clear weight

                #maybe dont need to update u_tilde for disc pts! since they wont show up in idxsI
                #(test later)
                if cinfo is not None:
                    cinfo['u_tilde'][pti] = 1/FN_D1s[pti] if degrees[pti] else 0

            
            if extraVerbose and xi == to_be_pruned[0]:
                print(f"{xi}'s {edge_tup} (stat={stati:.2f}) to ",end= '')
            
            #edge_idx = edges_idx_dict[edge_tup]
            pruned_edge_tups.append(edge_tup)


            nodes_seen.add(xi)
            nodes_seen.add(fnbr)

        if extraVerbose:
            print(f"{xi}'s {edge_tup} (stat={stati:.2f})")

        return (pruned_edge_tups,new_disc_pts,wA,sorted_nbrs,degrees,FN_D1s,non_nnd1s,sum_weights,cinfo)

    def plotGraphConnections(f, ax, Xplot, wA, title, it, pruned_edge_tups=[],
                              debugStatsPts=[], sigmas=None, scl=None, 
                              saveImgsto=None, img_format='pdf', dpi=150):
        if Xplot is None:
            return
        plotDataGraph(Xplot,wA,f_ax=(f,ax))
        ax.set_title(title)
        if pruned_edge_tups:
            lines = LineCollection([[(Xplot[xi,0],Xplot[xi,1]) for xi in e] for e in pruned_edge_tups],
                colors=[1,0,0,.8], #pruned edges in red 
                linewidths=1.5,
                zorder=-1)
            ax.add_collection(lines)
        if debugStatsPts:
            for xi in debugStatsPts:
                #plot scales
                circle = mpatches.Circle(Xplot[xi,:2], sigmas[xi]/scl,
                ls='-',lw=1,fill=False,label=str(xi))
                ax.add_patch(circle)
            ax.legend()

        if saveImgsto:
            plt.savefig(f'{saveImgsto}/it{it+1}.{img_format}',dpi=dpi)

    
    Adj, D1, D2, scl = process_input(X,Adj,method,metric,obj,max_nbrhood_size)
    if extraVerbose: print('Done processing input.')
    if return_processed_inputs:
        return Adj, D1/scl

    wA, sorted_nbrs, degrees, FN_D1s, sum_weights, non_nnd1s, med_nbr_ds = get_connectivity_data(Adj, D1, obj, optC is None)
    if excessiveVerbose: print('Done get_connectivity_data')

    if pre_deleted_nodes is not None:
        disc_pts, disc_pt_info = list(map(list,zip(*pre_deleted_nodes)))
    else:
        disc_pts, disc_pt_info = [], []

    pruning_history = []
    sigma_history = []


    itstart = time.time()
    it = 0

    total_pruned = 0


    if optC is None:
        optC = min( max( 0.55, np.median(med_nbr_ds/FN_D1s) ), .95)
        assert optC <= 1
        if extraVerbose:
            print('Initial C value estimated from data:', optC)
        C_tuning = True

    else:
        if optC > 1:
            raise ValueError('C must be positive <= 1 or None (auto-tuning)')
        if minimalVerbose:
            print(f'C value provided by user = {optC} -- skipping tuning.')
        C_tuning = False

    #setting slack for how close to 1 the median of the delta_i' distribution should be
    #  (results are quite robust to this choice, and making this tighter increases computation cost of C-tuning)
    median_tol = .1

    #if not C_tuning or obj == 'greedy':
    #cinfo carries along the constraints information -- this allows for a faster update of sigmas after each iteration
    #however, when nusing greedy approach, these are not needed, but still required as argument to pruning function
    cinfo = None

    if plot_interval > 0:
        n_subplots = 1 + int(allowMSconvergence) + int(Xplot is not None) #1 to 3 subplots, depending on arguments


    while it < maxIter:


        stats_args = dict(degrees=degrees, D2=D2, sig2scl=sig2scl, disc_pts=disc_pts, debugPts=debugStatsPts)

        optC, sigmas, stats, mu, cinfo, _ = getSigmasTuneC(C_tuning, optC, cinfo, wA, FN_D1s, obj, sum_weights, stats_args, 'median', 
                                    median_tol=median_tol, solver=solver, verbose=excessiveVerbose, optverbose=optverbose)
        
        diffmu = mu - 1

        sigma_history.append((optC,sigmas/scl))
        

        #(disconnected pts will never be pruned since they get 0 stats)
        for dpt in disc_pts:
            assert stats[dpt] <= 0

        for xi in debugStatsPts:
            print(f'{xi} stat:',stats[xi])
        nonz_stats = stats[stats > 0]


        ### plotting ###
        if plot_interval:
            thisPlot = it % plot_interval == 0
            if thisPlot:
                f, axes = subps(1,n_subplots,3.5,5,axlist=True)
                ax = axes[0]
        else:
            thisPlot = False
            ax = None


        ### compute adaptive threshold for current distribution
        mu_,stdev_,thresh,_ = computeThreshold(nonz_stats, nstds, 5 - diffmu, plot=thisPlot, ax=ax, stdev_method=stdev_method,
                                                plotQuartiles=plotQuartiles, title='Discrete graph statistics')

        #TODO:? sort stats and indices to be pruned together: sorted_stats, to_be_pruned = zip(stats, np.arange(N))
        #sorted_stats, idxs = zip(*sorted(zip(stats, np.arange(N), reverse=True)))


        to_be_pruned = np.flatnonzero( stats > thresh )
        #sort by stat value in decreasing order
        to_be_pruned = sorted(to_be_pruned, key=lambda i: stats[i], reverse=True)



        if extraVerbose:
            print(len(to_be_pruned),'nodes above threshold.')

        if len(to_be_pruned) == 0:
            #check that median (from C tuning) is approx 1
            if diffmu > median_tol:
                if extraVerbose:
                    print(f'! Distribution is not yet centered (median = {mu:.2f} >> 1). Keep pruning...')
                n_below = (nonz_stats < 1 + median_tol).sum()
                n_to_prune = nonz_stats.size - 2*n_below
                #add all those points preventing 1 from being the actual median
                to_be_pruned = np.argsort(stats)[-n_to_prune:][::-1]

            else:

                if minimalVerbose:
                    print('CONVERGED: no change in discrete graph.')
                    print('Total # edges pruned:',sum(map(len,pruning_history)))

                pruning_history.append([])

                #clear remaining subplot
                if allowMSconvergence and thisPlot:
                    axes[1].axis('off')

                break

        assert len(to_be_pruned) > 0

        
        if allowMSconvergence:
            # test for early convergence based on multi-scale weighted graph

            wG = getSparseMultiScaleK(**stats_args, optSigs=sigmas, verbose=excessiveVerbose,
                                            returnSparse=False, symmetrize=False)
            ms_stats = getVolumeRatios(**stats_args, wG=wG, sigmas=sigmas)

            ax = None if not thisPlot else axes[1]

            ms_mu,ms_stdev,ms_thresh,_ = computeThreshold(ms_stats[ms_stats > 0], nstds, 3 - diffmu, plot=thisPlot, ax=ax,
                stdev_method=stdev_method, plotQuartiles=plotQuartiles, title='Weighted graph statistics')

            ms_n_to_be_pruned = (ms_stats > ms_thresh).sum()

            if extraVerbose:
                print(f'{ms_n_to_be_pruned} nodes above ms_thresh.')

            if ms_n_to_be_pruned == 0:
                if minimalVerbose:
                    print('CONVERGED: weighted graph has no outliers.')
                    print('Total # edges pruned:',sum(map(len,pruning_history)))

                pruning_history.append([])

                break                    

        
        sorted_stats = sorted(stats[to_be_pruned],reverse=True)
        

        if maxPruneAtaTime is not None:
            #check max # edges allowed to be pruned per iter
            if type(maxPruneAtaTime) == int: #actual number (int)
                maxNPruned = maxPruneAtaTime

            elif type(maxPruneAtaTime) == float: #percentage
                maxNPruned = max(1,int(maxPruneAtaTime*len(to_be_pruned)))

            to_be_pruned = to_be_pruned[:maxNPruned]

        ### delete the most costly edge from each node to be pruned
        (pruned_edge_tups, new_disc_pts,
            wA,sorted_nbrs, degrees, FN_D1s,
            non_nnd1s, sum_weights, cinfo) = prune_edges(to_be_pruned, wA, sorted_nbrs, degrees, FN_D1s,
                                                    non_nnd1s, sum_weights, sorted_stats, cinfo)

        pruning_history.append(pruned_edge_tups)

        ### plotting the data graph, if points have been provided
        if thisPlot:
            if Xplot is not None:
                ax = axes[1 + int(allowMSconvergence)]
                plotGraphConnections(f, ax, Xplot, wA, f'Iteration {it+1}', it, pruned_edge_tups,
                                        debugStatsPts, sigmas, scl, **saveImgs)
            plt.show()


        if new_disc_pts:
            for dpt,dpt_nbr in new_disc_pts:
                #save last sigma, nbr before disconnection
                disc_pts.append(dpt)
                disc_pt_info.append( {'last_sig':sigmas[dpt]/scl,'last_nbr':dpt_nbr} )
                

        if minimalVerbose:
            print(f'### Iteration {it} done. ({time.time()-itstart:.2f} s) - pruned {len(to_be_pruned)} edge(s)')
        if extraVerbose:
            total_pruned += len(pruned_edge_tups)
            print(f'Total pruned so far: {total_pruned}', end=' ')
            if disc_pts:
                print(f'({len(disc_pts)} disconnected pts)')
            else:
                print()

        itstart = time.time()
        it += 1
        
        if it == maxIter:
            doBreak = True
            if interactive:
                if input("Reached max # iterations. Continue? Y or N").lower() == 'y':
                    doBreak = False
                    maxIter += 100
                    print("Changing max # iters to", maxIter)
            if doBreak:
                if minimalVerbose: print('STOPPING - Reached max iters')
                break

    ### end while

    ### plotting the data graph, after convergence
    if it < maxIter and thisPlot and Xplot is not None:
        ax = axes[1 + int(allowMSconvergence)]
        plotGraphConnections(f, ax, Xplot, wA, 'Converged.', it, [], debugStatsPts, sigmas, scl, **saveImgs)
        plt.show()

                    
    if 'approximate' in method:
        assert sp.issparse(D2)
    #     #convert D2 into N-by-N sparse matrix
    #     rows,cols,d1s = sp.sparse.find(D1)
    #     D2 = sp.sparse.csr_matrix((d1s**2,(rows,cols)))
    
    stats_args = dict(degrees=degrees, D2=D2, sig2scl=sig2scl, disc_pts=disc_pts)

    if tune_wG_method is not None:
        if minimalVerbose:
            print('C-tuning weighted graph...')        
        optC, optSigs, wstats, wmu, cinfo, wG = getSigmasTuneC(C_tuning, optC, cinfo, wA, FN_D1s, obj, sum_weights, stats_args, tune_wG_method, 
            use_wG=True, median_tol=median_tol, solver=solver, verbose=excessiveVerbose, optverbose=optverbose)

    else:
        if minimalVerbose:
            print('Computing weighted graph...')   
        optSigs = sigmas.copy()
        wG = getSparseMultiScaleK(**stats_args, optSigs=optSigs, verbose=excessiveVerbose)
        wstats = getVolumeRatios(**stats_args, wG=wG, sigmas=optSigs)
        
    
    if minimalVerbose:
        print(f'Discrete graph stats: mean={np.mean(nonz_stats):.3f}, median={np.median(nonz_stats):.3f}')
        final_nonz_wstats = wstats[wstats > 0]
        print(f'Weighted graph stats: mean={np.mean(final_nonz_wstats):.3f}, median={np.median(final_nonz_wstats):.3f}')

    if plot_final_stats:
        f,axes = subps(1,2 + int(Xplot is not None),3.5,5)
        mu,stdev,_,_ = computeThreshold(stats[stats > 0], nstds, plot=True, ax=axes[0],
            plotQuartiles=plotQuartiles, showThresh=False, title='Final volume ratios for discrete graph')

        mu,stdev,_,_ = computeThreshold(final_nonz_wstats, nstds, plot=True, ax=axes[1],
            plotQuartiles=plotQuartiles, showThresh=False, title='Final volume ratios for weighted graph')

        if Xplot is not None:
            plotGraphConnections(f, axes[2], Xplot, wA, r'Final $G$', it, **saveImgs)

        plt.show()


    #convert wA back to a sparse, N-by-N matrix (i.e., our final unweighted graph, G)
    rows, cols = list(zip(*wA.keys()))
    G = sp.sparse.csr_matrix((np.ones(2*len(wA)),(rows+cols,cols+rows)),shape=D1.shape)

    optSigs /= scl
    disc_pts = list(zip(disc_pts,disc_pt_info))
    
    if return_stats:
        return G, wG, optSigs, disc_pts, pruning_history, sigma_history, stats, wstats
    
    return G, wG, optSigs, disc_pts




def getSigmasFixedC(curr_C, cinfo, weighted_edges, FN_D1s, obj, stats_args, mu_method, use_wG, solver, verbose, optverbose):

    if obj == 'greedy':
        curr_sigmas = curr_C*greedySplitting(weighted_edges, FN_D1s.astype('float32'), 1, verbose=int(optverbose))
    else:
        if cinfo is None: #initialize constraints
            cinfo = getParamConstraintFixedC(weighted_edges, curr_C, FN_D1s)
        curr_sigmas = getSigmasFromConstraints(FN_D1s, cinfo, obj, solver, verbose, optverbose)

    assert curr_sigmas is not None

    curr_wG = None
    if use_wG:
        curr_wG = getSparseMultiScaleK(**stats_args,optSigs=curr_sigmas,verbose=verbose)
    curr_ratios = getVolumeRatios(**stats_args,wG=curr_wG,sigmas=curr_sigmas)

    #given the new volume ratios, update median of distribution
    mu = getMuStdev(curr_ratios[curr_ratios > 0],mu_method)
    

    if verbose: print(f'Fixed C = {curr_C} -- recomputed sigmas. New mu={mu:.8f}')


    return curr_C, curr_sigmas, curr_ratios, mu, cinfo, curr_wG


def getSigmasTuneC(C_tuning, curr_C, cinfo, weighted_edges, FN_D1s, obj, sum_weights, stats_args, mu_method, use_wG=False, median_tol=.05,
                    solver=None, verbose=False, optverbose=False):


    ###DEBUGGIN - CASES TO TEST
    #a1 - first iteration, C_tuning
    #a2 - first iteration, no C_tuning
    #a3 - first iteration, greedy

    #b1 - i-th iter, C not changing (C_tuning)
    #b2 - i-th iter, C fixed
    #b3 - i-th iter, C not changing (greedy)

    #c1 - i-th iter, C changing (C_tuning)
    #c2 - i-th iter, C changing (greedy)

    #d1 - ms-tuning
    #d2 - wG tuning

    if not C_tuning:
        return getSigmasFixedC(curr_C, cinfo, weighted_edges, FN_D1s, obj, stats_args, mu_method, use_wG, solver, verbose, optverbose)
    
    minC = min(curr_C,0.5)
    maxC = max(curr_C,1)
    curr_wG = None

    def convergedC():
        median_close_to_1 = abs(diffmu) <= median_tol
        return  median_close_to_1 or \
             ( diffmu > median_tol and np.isclose(curr_C, minC) ) or \
             ( diffmu < -median_tol and np.isclose(curr_C, maxC) )

    # 1) recompute sigmas using current C -- they will have changed after the last pruning step
    #    then check if we will actually need to retune C
    if cinfo is not None or obj == 'greedy': #means obj is either greedy, or iter > 0
        # if cinfo is None (meaning 1st iteration), we skip this and use the more efficient optimization below (part 2),
        #    since C will certainly need tuning

        if obj == 'greedy':
            #defining updating function to be possibly used again in part 2
            base_sigmas = greedySplitting(weighted_edges, FN_D1s, 1, verbose=int(optverbose))
            def getNewSigmas(Cval):
                return Cval*base_sigmas

            curr_sigmas = getNewSigmas(curr_C)

        else:
            if verbose: print('+++Recycling constraints')
            #recycle pre-built constraints, for speed
            curr_sigmas = getSigmasFromConstraints(FN_D1s, cinfo, obj, solver, verbose, optverbose)
            
        assert curr_sigmas is not None

        if use_wG:
            curr_wG = getSparseMultiScaleK(**stats_args,optSigs=curr_sigmas,verbose=verbose)

        curr_ratios = getVolumeRatios(**stats_args,wG=curr_wG,sigmas=curr_sigmas)

        #given the new volume ratios, check how close the sample median is to 1
        mu = getMuStdev(curr_ratios[curr_ratios > 0],mu_method)
        if verbose: print(f'First pass: curr_C={curr_C:.8f}, mu={mu:.8f}')

        diffmu = mu - 1
        if convergedC():
            if verbose: print(f'Not re-tuning: diff_mu = {abs(diffmu):.8f}, curr_C = {curr_C:.8f}')
            return curr_C, curr_sigmas, curr_ratios, mu, cinfo, curr_wG

        # if no immediate convergence, need to retune C
        # get first update to current C from this initial pass
        if diffmu > 0: #median > 1
            maxC = curr_C
        else: #median < 1
            minC = curr_C
        #bisect
        curr_C = minC + .5*(maxC - minC)

    # 2) reach here if C needs to be (re)-computed

    if obj != 'greedy':
        # define constaints using C as a variable parameter
        # (because C changes, the constraints need to be re-computed from scratch)
        # this method allows for re-optimizing saving lots of time
        x, objective, constraints, C, C_tilde = buildOptimizationProblem(weighted_edges, FN_D1s, objfoo=obj, 
                                                                            sum_weights=sum_weights, verbose=verbose)
        def getNewSigmas(Cval):
            C.value = Cval
            C_tilde.value = 1./C.value
            prob = cp.Problem(cp.Minimize(objective), constraints)
            result = prob.solve(solver=solver,
                verbose=bool(optverbose),**solver_opts.get(solver,{}))
            if verbose: print('Solved sigmas. Obj =',result)
            return x.value

    # bisection search for optimal C 
    # (up to a prespecified max # of iters  -- no need to be too precise here since this 
    #   will be reassessed after each pruning) 
    nits = 0

    while nits < 20: #20 should be more than enough to guarantee convergence
        
        curr_sigmas = getNewSigmas(curr_C)
        assert curr_sigmas is not None

        if use_wG:
            curr_wG = getSparseMultiScaleK(**stats_args, optSigs=curr_sigmas, verbose=verbose)

        curr_ratios = getVolumeRatios(**stats_args, wG=curr_wG, sigmas=curr_sigmas)
        mu = getMuStdev(curr_ratios[curr_ratios > 0],mu_method)
        if verbose: print(f'{nits}: curr_C={curr_C:.8f}, mu={mu:.8f}')

        diffmu = mu - 1
        if convergedC():
            if verbose: print(f'Converged: diff_mu = {abs(diffmu):.8f}, curr_C = {curr_C:.8f}')
            #converged
            break

        #get first update to C from this initial pass
        if diffmu > 0: #median > 1
            maxC = curr_C
        else:
            minC = curr_C
        curr_C = minC + .5*(maxC - minC)
        nits += 1

    if obj != 'greedy':
        # (re)-initialize constraints for next iters
        cinfo = getParamConstraintFixedC(weighted_edges, curr_C, FN_D1s)


    return curr_C, curr_sigmas, curr_ratios, mu, cinfo, curr_wG


def getParamConstraint(weighted_edges,u):
    b1 = []
    b2 = []

    slopes = []
    
    C1 = []
    C_1 = []

    b1Cs = []
    C2 = []
    C0 = []
    idxsI = []
    idxsJ = []
    for (xi,xj),e_len in weighted_edges.items():

        idxsI += [xi,xi]
        idxsJ += [xj,xj]

        #RHS secant
        slopes.append(-e_len/u[xi])
        
        C1.append(1)
        C_1.append(0)
        b1.append(-e_len**2/u[xi])
        C2.append(1)
        C0.append(0)
        b2.append(-e_len)

        #LHS secant
        slopes.append(-u[xj]/e_len)
        
        C_1.append(1)
        C1.append(0)
        b1.append(-u[xj])
        C2.append(0)
        C0.append(1)
        b2.append(-e_len)

    slopes = np.array(slopes)
    C1 = np.array(C1)
    C_1 = np.array(C_1)
    b1 = np.array(b1)
    C2 = np.array(C2)
    C0 = np.array(C0)
    b2 = np.array(b2)
    return idxsI, idxsJ, slopes, C1, C_1, b1, C2, C0, b2

def getParamConstraintFixedC(weighted_edges,C,FN_D1s):
    b1 = []
    b2 = []

    slopes = []
    slopeCs = []
    C1 = []
    C_1 = []

    b1Cs = []
    C2 = []
    C0 = []
    idxsI = []
    idxsJ = []
    idxsU_1, idxsU = [], []
    u_1s, us = [], []

    edge_is = {}
    ei = 0
    for (xi,xj),e_len in weighted_edges.items():
        
        w = e_len*C

        idxsI += [xi,xi]
        idxsJ += [xj,xj]
#         idxsU_1 += [xi,xi]
#         idxsU += [xj,xj]

        #RHS secant
        slopes.append(-w)
        u_1s.append(1)
        us.append(0)
        b1.append(-w**2)
        b2.append(-w)

        #LHS secant
        slopes.append(-1/w)
        u_1s.append(0)
        us.append(1)
        b1.append(-1)
        b2.append(-w)

        edge_is[(xi,xj)] = [ei,ei+1]

        ei += 2


    idxsI = np.array(idxsI)
    idxsJ = np.array(idxsJ)
    slopes = np.array(slopes)
    b1 = np.array(b1)
    b2 = np.array(b2)
    u_1s = np.array(u_1s)
    us = np.array(us)

    e_idxs = np.arange(len(slopes),dtype='int32')

    u_tilde = FN_D1s.copy()
    u_tilde[u_tilde>0] = 1/u_tilde[u_tilde>0]

    cinfo = {'edge_is':edge_is, 'idxsI':idxsI, 'idxsJ':idxsJ, 'slopes':slopes,
                'b1':b1, 'b2':b2, 'us':us, 'u_1s':u_1s, 'e_idxs':e_idxs, 'u_tilde':u_tilde}

    return cinfo


def buildOptimizationProblem(weighted_edges,D1_fn,objfoo='sum',sum_weights=None,verbose=False):

    if verbose:
        print('Building constraints...',end=' ')
        start = time.time()
    n = len(D1_fn)

    C = cp.Parameter(nonneg=True)
    C_tilde = cp.Parameter(nonneg=True)
    x = cp.Variable(n)

    idxsI, idxsJ, slopes, C1, C_1, b1, C2, C0, b2 = getParamConstraint(weighted_edges,D1_fn)
    
    m = slopes.shape[0]

    slopesC1 = slopes*C1
    slopesC_1 = slopes*C_1
    
    addCs = C*slopesC1 + C_tilde*slopesC_1

    constraints = [ cp.multiply(addCs,x[idxsI]) - x[idxsJ] <= cp.multiply(b1,(C**2)*C2+C0) + b2*C ]

    # if solver == 'GUROBI-bounds':
    #     lb = np.zeros(n)
    #     ub = D1_fn.ravel()
    # else:
    A_ = sp.sparse.vstack([sp.sparse.identity(n),-sp.sparse.identity(n)])
    b_ = sp.sparse.csc_matrix((D1_fn.ravel(),(range(n),[0]*n)),shape=(2*n,1))

    constraints.append( A_ @ x[:,None] <= b_  )

    if verbose: print(f'done. ({time.time()-start:.1}s), {m} linear constraints')

    if objfoo in ['wsum','non']:
        assert sum_weights is not None
        c = sum_weights
    elif objfoo == 'sum':
        c = np.ones(n)
    else:
        raise ValueError('objective function not recognized')

    objective = c @ x
    prob = cp.Problem(cp.Minimize(objective), constraints)
    assert prob.is_dcp(dpp=True)
    for cons in constraints:
        assert cons.is_dcp(dpp=True)
        
    return x,objective,constraints,C,C_tilde


############### NCD ##################



def computeNofNsDims(NofNs_dict,data=None,cellNbrs=None,bounds=None,boundtol=1e-4,deltalog=1e-2,computeExact=False,
                         computeAvgsFromCurve=False,verbose=True,special_points=[],subset=None,highestPeaks=True,returnSigs=False):
    def findPeak(a,n=0):
        maxs = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
        assert len(maxs) >= n+1
        return maxs[n]

    def getMark(g2,startn=0,g3=None,returnAll=False):

        #use the first max of g2 as starting point to escape numerical errors at the start of the curves
        starti = findPeak(g2,startn)

        #TODO: try to find closed forms for the zeros g2, g3

        #g2 0-x
        zero_cross = np.flatnonzero( (g2[starti:-1] > 0) & (g2[starti+1:] < 0) )
        assert len(zero_cross) >= 1
        if returnAll:
            g2x0 = zero_cross + starti
        else:
            g2x0 = zero_cross[0] + starti
    #     #can do the same looking for first peak in g1 -- less precise b/c with 0-x we interpolate

    #     maxs = np.flatnonzero( (g1[starti+1:-1] > g1[starti:-2] + tol) & (g1[starti+1:-1] > g1[starti++2:] + tol) ) + 1
    #     assert len(maxs) >= 1
    #     g1max = maxs[0] + starti
    #     assert g1max - g2x0 <= 1

        #TODO: use 0x of G2 with a scipy root-finding algorithm in the interval [g2x0,g2x0+1]


        #g3 0-x
        if g3 is not None:
            g3zero_cross = np.flatnonzero( (g3[starti:-1] < 0) & (g3[starti+1:] > 0) )
            assert len(g3zero_cross) >= 1
            if returnAll:
                g3x0 = g3zero_cross + starti
            else: g3x0 = g3zero_cross[0] + starti
            g2min = g3x0
        else:
    #     #can do the same looking for first trough in g2
            mins = np.flatnonzero( (g2[starti+1:-1] < g2[starti:-2]) & (g2[starti+1:-1] < g2[starti++2:]) ) + 1
            assert len(mins) >= 1
            if returnAll:
                g2min = mins + starti
            else: g2min = mins[0] + starti
    #     assert g2min - g3x0 <= 1

        #TODO: use 0x of G3 with a scipy root-finding algorithm in the interval [g2min-1,g2min+1]

        return g2x0,g2min
    def computeCellBounds(ds,boundtol = 1e-4):

        #compute left lim using closest nbr
        llim = 0; lval = np.inf
        myd = ds[ds > 0].min()
        while lval > boundtol:
            llim -= .25
            lval = np.log10(1 + np.exp(-myd/(2*(10**llim)**2)))
        #compute right lim using furthest nbr
        rlim = 0; rval = 0
        myd = ds.max()
        while np.log10(2) - rval > boundtol:
            rlim += .25
            rval = np.log10(1 + np.exp(-myd/(2*(10**rlim)**2)))

        return llim,rlim
    def getG12(returnSumExps=False,returnG3=False):
        sumexps = np.sum(exps,axis=0)
        sumdsexps = np.sum(dsexps,axis=0)
        sumd2sexps = np.sum(d2sexps,axis=0)

        den = sumexps * s2
        num = sumdsexps
        g1 = num/den

        #g2 - closed form
        num0 = sumd2sexps
        num1 = num**2
        num2 = 2*num

        s4 = s2*s2
        den0 = sumexps * s4
        den1 = sumexps * den0
        den2 = den

        g2 = (num0/den0 - num1/den1 - num2/den2)*np.log(10)
        
        
        if not returnG3:       
            if returnSumExps:
                return g1,g2,sumexps,sumdsexps,sumd2sexps
            return g1,g2
        
        # g3 - closed form
        sumd3sexps = np.sum(d3sexps,axis=0)
        s3 = s2*sigmas
        s7 = s4*s3
        sumexps2 = sumexps**2
        sumexps3 = sumexps2*sumexps
        s6 = s4*s2
        num_2 = num**2
        num_3 = num_2*num

        num00 = sumd3sexps
        den00 = sumexps*s6
        num01 = -4*num0
        den01 = den0
        num02 = -num*num0
        den02 = sumexps2*s6

        g3_0 = num00/den00 + num01/den01 + num02/den02


        num10 = -2*num0*num
        den10 = sumexps2*s6
        num11 = 4*num_2
        den11 = sumexps2*s4
        num12 = 2*num_3
        den12 = sumexps3*s6

        g3_1 = num10/den10 + num11/den11 + num12/den12


        num20 = -2*num0
        den20 = sumexps*s4
        num21 = 2*num_2
        den21 = sumexps2*s4
        num22 = 4*num
        den22 = sumexps*s2

        g3_2 = num20/den20 + num21/den21 + num22/den22

        g3 = (g3_0 + g3_1 + g3_2)*np.log(10)**2
        
        if returnSumExps:
            return g1,g2,g3,sumexps,sumdsexps,sumd2sexps,sumd3sexps
        return g1,g2,g3
    

        
    N = len(NofNs_dict)
    if subset is None:
        subset = range(N)

    cellNofNs = {}
    if returnSigs:
        NofNSigs = np.zeros(N)
    NofNDims = np.zeros(N)


    if bounds is None:    
        bounds = np.zeros((N,2))
        for c in subset:#range(N):
            bounds[c] = computeCellBounds(NofNs_dict[c]['D2'],boundtol)
    llim, rlim = bounds[:,0].min(),bounds[:,1].max()

    lsigmas = np.arange(llim,rlim+deltalog,deltalog)
    sigmas = 10**(lsigmas)
    s2 = sigmas**2
    ns = len(sigmas)

    
    #compute curves for n-of-ns
    for ci,c in enumerate(subset):#range(N):
        if verbose and ci % 250 == 0: print(ci,end=' ')
        NofNs = NofNs_dict[c]['NofNs']


        ds = NofNs_dict[c]['D2'] # each xi might have different no. of nbrs & n-of-nbrs
        exps = np.array([np.exp(-d2/(2*s2)) for d2 in ds])
        dsexps = ds[:,None] * exps
        d2sexps = ds[:,None] * dsexps
        
        
        # N-of-Ns curve
        if computeExact:
            d3sexps = ds[:,None] * d2sexps
            nofn_g1,nofn_g2,nofn_g3 = getG12(returnG3=True)
        else:
            nofn_g1,nofn_g2 = getG12()
            nofn_g3 = None

        g2x0,g3x0 = getMark(nofn_g2,g3=nofn_g3,returnAll=highestPeaks)

    
            
        if highestPeaks:
            g2x0 = g2x0[np.argmax(nofn_g1[g2x0])]
            g3x0 = np.inf#g3x0[np.argmax(nofn_g1[g3x0])] ##must be an actual max


            
        #x0 = g3x0 if g3x0 < g2x0 else g2x0
        
        if computeExact:        

            allNNds = NofNs_dict[c]['D2']

            if g3x0 < g2x0:
                s_peak = brentq(g3exact,sigmas[g3x0],sigmas[g3x0+1],(allNNds))
                nofn_x0 = g3x0

            else:
                s_peak = brentq(g2exact,sigmas[g2x0],sigmas[g2x0+1],(allNNds))
                nofn_x0 = g2x0
                
            nofn_sig = s_peak
            nofn_dim = g1exact(s_peak,allNNds)
        else:
            nofn_x0 = g3x0 if g3x0 < g2x0 else g2x0
        
            nofn_sig = sigmas[nofn_x0]
            nofn_dim = nofn_g1[nofn_x0]
        if returnSigs:
            NofNSigs[c] = nofn_sig
        NofNDims[c] = nofn_dim

        ###PLOTS
        if c in special_points:
            print('c',c)
            if cellNbrs is not None:
                NNs = list(set(cellNbrs[c]).difference([c]))
                print('NNs',NNs)
            print('NofNs',NofNs)
            print('d2s',ds)



            plt.figure(figsize=(10,4))
            ax = plt.subplot(121)
            ax.set(title=f'$x_{{{c}}}$ - G1 curve',xlabel='s',ylabel='dim')
            for lbl,g1,sig,x0,dim,color in [
                                     ('nbrs-of-nbrs',nofn_g1,nofn_sig,nofn_x0,nofn_dim,'m'),
                                     ]:
                ax.plot(lsigmas,g1,c=color,label=lbl)
                #ax.plot([np.log10(sig),np.log10(sig)],[0,g1[x0]],c=color,ls=':',lw=2,label=f'$\sigma$={sig:.2f}')
                ax.plot([lsigmas[0],np.log10(sig)],[g1[x0],g1[x0]],c=color,ls=':',lw=2,label=f'$d$={dim:.2f}')
                if computeExact and lbl == 'nbrs-of-nbrs':
                    ax.plot(lsigmas,[g1exact(s,allNNds) for s in sigmas],'r-.',lw=3,label='nofnX')
            ax.legend()
            
            #SHOW DATA POINTS
            ax = plt.subplot(122)
            ax.scatter(*data[:,:2].T,s=3)
            ax.scatter(*data[c:c+1,:2].T,color='r',marker='o',facecolor='none',s=40)

            if cellNbrs is not None:
                ax.scatter(*data[NNs,:2].T,color='g',marker='o',facecolor='none',s=40)
            ax.scatter(*data[NofNs,:2].T,color='slategray',marker='o',facecolor='none',s=120)

            ax.axis('equal')


            plt.show()
            print()
    if returnSigs:
        return NofNDims, NofNSigs

    return NofNDims

def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = nbrs.union([nbr for n in nbrs for nbr in G[n]])
    return nbrs


def estimateDims(A, D2, nbrhoodOrder, centralPtDim=True, nbrsAvgDims=True, useMedian=True, useDegDims=True,
    debugPts=[], X=None, verbose=0):
    
    def getRecenteredHood(xi,NofNs,pwdists2,useMedian=True,debug=False,X=None):
        hood = NofNs[xi]
        n = len(hood)
        assert type(hood) == list
        if useMedian:
            dists_stats  = np.empty(n,dtype='d')
            for ii,xj in enumerate(hood):
                dists_stats[ii] = np.median(pwdists2[xj,hood])

    #         dists_stats_ = pwdists2[hood][:,hood].sum(1)
    #         amin,amin_ = dists_stats.argmin(), dists_stats_.argmin()
    #         if amin != amin_:
    #             print(xi,'different argmins',hood[amin],hood[amin_])
        else:
            dists_stats = pwdists2[hood][:,hood].sum(1)
        center_node = hood[dists_stats.argmin()]

        #NEW: re-center and add center_node's nbrhood to original hood
        xiHood = list(set(hood + NofNs[center_node]))
        xiD2 = pwdists2[center_node][xiHood] 
        if debug:
            print('xi',xi)
            print('original hood',sorted(hood))

            print('center_node',center_node)
            print('new hood',sorted(xiHood))
            print('xiD2',xiD2)
            print('Compare with')
            print(pwdists2[xi][NofNs[xi]])
            if X is not None:
                ax = plt.subplot(111)
                ax.scatter(*X[:,:2].T,s=3)
                ax.scatter(*X[xi:xi+1,:2].T,color='r',marker='*',facecolor='none',s=40)
                ax.scatter(*X[center_node:center_node+1,:2].T,color='y',marker='o',facecolor='none',s=40)
                if useMedian:
                    sum_center = hood[pwdists2[hood][:,hood].sum(1).argmin()]
                    ax.scatter(*X[sum_center:sum_center+1,:2].T,color='k',marker='o',facecolor='none',s=40)
                ax.scatter(*X[NofNs[xi],:2].T,color='slategray',marker='o',facecolor='none',s=120)
                xtra = list(set(xiHood).difference(NofNs[xi]))
                if xtra:
                    ax.scatter(*X[xtra,:2].T,color='g',marker='o',facecolor='none',s=120)
                ax.axis('equal')
                plt.show()
        return xiD2,xiHood,center_node
    
    nbrs_dict = {}
    N = A.shape[0]
    if sp.sparse.issparse(A):
        A = A.tocsr()
    if useDegDims:
        deg_dims = np.zeros(N)
    for xi in range(N):
        if sp.sparse.issparse(A):
            nbrs_dict[xi] = A[xi].indices
        else:
            nbrs_dict[xi] = np.flatnonzero(A[xi])
        if useDegDims:
            deg_dims[xi] = max(1,np.log2(len(nbrs_dict[xi])))

    NofNs = {xi:list(knbrs(nbrs_dict,xi,nbrhoodOrder)) for xi in range(N)}

    NofNs_dict = {xi:{}  for xi in range(N)}


    for xi in range(N):
        if verbose and xi % 100 == 0: print(xi,end=' ')

        if centralPtDim:
            debug = xi in debugPts
            xiD2,xiHood,center_node = getRecenteredHood(xi,NofNs,D2,useMedian,debug,X)


        else:
            xiD2 = D2[xi,NofNs[xi]]
            xiHood = NofNs[xi]
        NofNs_dict[xi]['D2'] = xiD2
        NofNs_dict[xi]['NofNs'] = xiHood
    NofNDims = computeNofNsDims(NofNs_dict,X,highestPeaks=True,special_points=debugPts,verbose=verbose)

    if nbrsAvgDims:
        nbrAvgDims = np.zeros(N)
        if useDegDims:
            avg_deg_dims = np.zeros(N)
        for xi in range(N):
            if len(nbrs_dict[xi]) == 0:
                continue

            nbrAvgDims[xi] = np.mean(NofNDims[nbrs_dict[xi]])
            if useDegDims:
                avg_deg_dims[xi] = np.mean(np.floor(deg_dims[nbrs_dict[xi]]))
                #nbrAvgDims[xi] = max(nbrAvgDims[xi], avg_dim_)
            
            if xi in debugPts:
                print('***',xi)
                print('nbrs:',nbrs_dict[xi])
                print('dims:',NofNDims[nbrs_dict[xi]])
                print('mean',nbrAvgDims[xi])
        NofNDims = nbrAvgDims
        if useDegDims:
            deg_dims = avg_deg_dims

    if useDegDims:
        return NofNDims, deg_dims
    return NofNDims


############# GEODESICS

def computeHeatGeodesics(K, t, chosen_pts, verbose=0):
    N = K.shape[0]
    grad = lambda u,W: np.sqrt(W) * (np.diag(u) @ np.ones((N,N)) - np.ones((N,N)) @ np.diag(u))
    div = lambda X_,W: np.sum(np.sqrt(W)*(X_.T - X_),1)[None,:]

    K_ = K.copy()
    np.fill_diagonal(K_,0)
    L = (K_ - np.diag(K_.sum(1)))


    u0 = np.zeros(N,)
    u0[chosen_pts] = 1

    areas = np.ones(N)
    Ac = np.eye(N)*areas
    Ac_ = np.eye(N)/areas


    Lc = Ac_ @ L
    u = np.linalg.solve(Ac-t*Lc,u0)
    if verbose: print('solved diffusion')

    Du = grad(u,K_)
    eps = np.finfo(K.dtype).eps
    norms = np.linalg.norm(Du,axis=1,keepdims=1)
    norms[norms < eps] = eps
    h = -Du / norms
    if verbose: print('solved gradients')

    mydiv = div(h,K_).ravel()

    phi = np.linalg.solve(Lc,mydiv)
    if verbose: print('solved div')
    phi -= phi.min()

    return phi


