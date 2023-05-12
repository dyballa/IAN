import numpy as np
import scipy as sp
import time
from scipy.spatial.distance import squareform

##### isomap
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import _fix_connected_components
from sklearn.manifold import smacof
from sklearn.decomposition import KernelPCA
def computeIsomap(X, knn,n_components, knbrs_graph=None, use_smacof=False, rs=0):
    """ Computes isomap from data points and a number of nearest neighbors, k, 
          or from a pre-specified unweighted graph """
    neigh = None
    if knbrs_graph is None:
        neigh = NearestNeighbors(algorithm='kd_tree').fit(X)
        knbrs_graph = kneighbors_graph(neigh, knn, mode='distance')
    n_connected_components, labels = connected_components(knbrs_graph)
    if n_connected_components > 1:

        # use array pre-validated by NearestNeighbors
        if neigh is None:
            neigh = NearestNeighbors(algorithm='kd_tree').fit(X)
            
        knbrs_graph = _fix_connected_components(
            neigh._fit_X,knbrs_graph.tolil(),n_connected_components,labels,mode="distance",
        )

    dist_matrix_ = shortest_path(knbrs_graph, directed=False)
    
    if use_smacof:
        embedding,*_ = smacof(dist_matrix_,random_state=rs)
    else:
        G = -.5 * np.square(dist_matrix_)

        kernel_pca_ = KernelPCA(
            n_components=n_components,
            kernel="precomputed",
        )
        embedding = kernel_pca_.fit_transform(G)#includes centering G
    
    return knbrs_graph, embedding

##### diffusion maps

def diffusionMapFromK(K, n_components, alpha=0, t=1, tol=1e-8, lambdaScale=True,
    returnPhi=False, returnOrtho=False, unitNorm=False, sparse_eigendecomp=True, use_svd=True):
    """Computes a diffusion map embedding from a kernel matrix"""

    if sp.sparse.issparse(K):
        return diffusionMapSparseK(K, n_components, alpha, t, lambdaScale,
                    returnPhi, returnOrtho, unitNorm, use_svd)
    else:
        return diffusionMapK(K, n_components, alpha, t, tol, lambdaScale,
                    returnPhi, returnOrtho, unitNorm, sparse_eigendecomp, use_svd)


def diffusionMapSparseK(K, n_components, alpha=0, t=1, lambdaScale=True, 
    returnPhi=False, returnOrtho=False, unitNorm=False, use_svd=True):
    """ alpha: 0 (markov), 1 (laplace-beltrami), 0.5 (fokker-plank) """
    assert alpha >= 0
    assert sp.sparse.issparse(K)
        
    N = K.shape[0]

    eps = np.finfo(K.dtype).eps

    #normalize (kernel)-adjacency matrix by each node's degree
    if alpha > 0:
        D = np.array(K.sum(axis=1)).ravel()
        Dalpha = np.power(D, -alpha)
        #right-normalize
        Dalpha = sp.sparse.spdiags(Dalpha, 0, N, N)
        K = Dalpha * K * Dalpha


    sqrtD = np.sqrt(np.array(K.sum(axis=1)).ravel()) + eps

    #symmetrizing Markov matrix by scaling rows and cols by 1/sqrt(D)
    sqrtDs = sp.sparse.spdiags(1/sqrtD, 0, N, N)


    Ms = sqrtDs * K * sqrtDs

    #ensure symmetric numerically
    Ms = Ms.maximum(Ms.transpose()).tocsr()

    SPARSETOL = 2*eps
    #bring out the zeros before eigendecomp (equiv. to: Ms[Ms < SPARSETOL] = 0)
    for i in range(N):
        Ms[i].data[Ms[i].data < SPARSETOL] = 0
    Ms.eliminate_zeros()

    k = n_components + 1

    assert k <= N, "sparse routines require n_components + 1 < N"

    if use_svd:
        U, lambdas, _ = sp.sparse.linalg.svds(Ms, k=k, return_singular_vectors='u')
    else:
        lambdas, U = sp.sparse.linalg.eigsh(Ms,k)#,overwrite_a=True,check_finite=False)


    #sort in decreasing order of evals
    idxs = lambdas.argsort()[::-1]
    lambdas = lambdas[idxs]
    U = U[:,idxs] #Phi

    if returnOrtho:
        Psi = U / U[:,0:1]
    elif returnPhi:
        assert sqrtD.ndim == 1 #assert col vector
        Psi = U * sqrtD[:,None]
    else:
        assert sqrtD.ndim == 1 #assert col vector
        Psi = U / sqrtD[:,None]
        #Phi = U * sqrtD
        #assert np.all(np.isclose((Phi.T @ Psi),np.eye(Psi.shape[1])))

    #make Psi vectors have unit norm
    if unitNorm:
        Psi = Psi / np.linalg.norm(Psi,axis=0,keepdims=1)

    if lambdaScale:
        assert (lambdas.ndim == 1 and lambdas.shape[0] == Psi.shape[1])
        if t == 0:
            diffmap = Psi * np.power(-1/(lambdas-1),.5)
        else:
            diffmap = Psi * np.power(lambdas,t)

    else: diffmap = Psi

    return diffmap[:,1:],lambdas[1:]

def diffusionMapK(K, n_components, alpha=0, t=1, tol=1e-8, lambdaScale=True,
    returnPhi=False, returnOrtho=False, unitNorm=False, sparse_eigendecomp=True, use_svd=True):
    """ alpha: 0 (markov), 1 (laplace-beltrami), 0.5 (fokker-plank) """

    assert alpha >= 0
    if sp.sparse.issparse(K):
        K = K.toarray()

    #assert symmetry
    assert np.all(np.isclose(K - K.T,0))

    N = K.shape[0]

    eps = np.finfo(K.dtype).eps

    #normalize (kernel) adjacency matrix by each node's degree
    if alpha > 0:
        #find degree q for each row
        D = K.sum(axis=1,keepdims=1) #always >= 1
        K = K / np.power(D @ D.T,alpha)

        
    #ignore kernel vals that are too small
    K[K < tol] = 0

    #symmetrizing Markov matrix by scaling rows and cols by 1/sqrt(D)
    sqrtD = np.sqrt(K.sum(axis=1,keepdims=1)) + eps #could be zero!
    Ms = K / (sqrtD @ sqrtD.T)

    #ensure symmetric numerically
    Ms = 0.5*(Ms + Ms.T)

    SPARSETOL = 2*eps
    Ms[Ms < SPARSETOL] = 0 #bring out the zeros before converting to sparse

    k = n_components + 1

    assert k <= N
    if k == N:
        sparse_eigendecomp = False #cannot compute all evals/evecs when sparse!

    if sparse_eigendecomp:
        sMs = sp.sparse.csc_matrix(Ms)
        if use_svd: #sparse svd
            U, lambdas, _ = sp.sparse.linalg.svds(sMs, k=k, return_singular_vectors='u')
        else:
            lambdas, U = sp.sparse.linalg.eigsh(sMs,k)#,overwrite_a=True,check_finite=False)
    else:
        if use_svd:
            U, lambdas, _ = np.linalg.svd(Ms, full_matrices=False)
        else:
            lambdas, U = np.linalg.eigh(Ms)


    #sort in decreasing order of eigenvalues
    idxs = lambdas.argsort()[::-1]
    lambdas = lambdas[idxs]
    U = U[:,idxs] #Phi
    
    if not sparse_eigendecomp:
        lambdas = lambdas[:k+1]
        U = U[:,:k+1]

    if returnOrtho:
        Psi = U / U[:,0:1]
    elif returnPhi:
        assert sqrtD.shape[1] == 1 #assert col vector
        Psi = U * sqrtD
    else:
        assert sqrtD.shape[1] == 1 #assert col vector
        Psi = U / sqrtD
        #Phi = U * sqrtD
        #assert np.all(np.isclose((Phi.T @ Psi),np.eye(Psi.shape[1])))

    #make Psi vectors have unit norm before scaling by eigenvalues
    if unitNorm:
        Psi = Psi / np.linalg.norm(Psi,axis=0,keepdims=1)

    if lambdaScale:
        assert (lambdas.ndim == 1 and lambdas.shape[0] == Psi.shape[1])
        diffmap = Psi * np.power(lambdas,t)
    else: diffmap = Psi

    return diffmap[:,1:],lambdas[1:]

###### t-SNE

from sklearn.manifold import TSNE
from ian.cutils import _binary_search_perplexity, get_tsne_sigmas
import warnings
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_non_negative, check_random_state

def my_joint_probabilities(sqdistances, desired_perplexity, precomputed_sigmas=None, verbose=False):
    """Compute joint probabilities p_ij from distances, and, 
     optionally, using precomputed sigmas (kernel scales).
     Code adapted from the scikit-learn package: https://scikit-learn.org/
     
     
    Parameters
    ----------
    sqdistances : square ndarray of shape (n_samples-by-n_samples) with squared distances
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : ndarray of shape (n_samples-by-n_samples)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    sqdistances = sqdistances.astype(np.float32, copy=False)
    if precomputed_sigmas is None:
        conditional_P = _binary_search_perplexity(
            sqdistances, desired_perplexity, int(verbose)
        )
    else:
        EPSILON_DBL = 1e-8
        
        #instead of optimizing sigmas based on perplexity, use pre-computed ones
        conditional_P = np.zeros_like(sqdistances)
        for i in range(conditional_P.shape[0]):
            conditional_P[i] = np.exp(-sqdistances[i]/(2*precomputed_sigmas[i]**2))
            conditional_P[i,i] = 0
            sum_Pi = max(conditional_P[i].sum(),EPSILON_DBL)
            conditional_P[i] /= sum_Pi
            
    np.testing.assert_allclose(conditional_P.sum(1),1,1e-5,1e-5)
            
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"
    
    MACHINE_EPSILON = np.finfo(np.double).eps

    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def my_joint_probabilities_nn(distances, desired_perplexity, precomputed_sigmas=None, verbose=False):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.
    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).
    Code adapted from the scikit-learn package: https://scikit-learn.org/
    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time.time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    if precomputed_sigmas is None:
        conditional_P = _binary_search_perplexity(
            distances_data, desired_perplexity, int(verbose)
        )
    else:
        EPSILON_DBL = 1e-8
        
        #instead of optimizing sigmas based on perplexity, use pre-computed ones
        conditional_P = np.zeros_like(distances_data)
        for i in range(conditional_P.shape[0]):
            conditional_P[i] = np.exp(-distances_data[i]/(2*precomputed_sigmas[i]**2))
            sum_Pi = max(conditional_P[i].sum(),EPSILON_DBL)
            conditional_P[i] /= sum_Pi      
            
    np.testing.assert_allclose(conditional_P.sum(1),1,1e-5,1e-5)

            
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T
    
    MACHINE_EPSILON = np.finfo(np.double).eps

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time.time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P

def my_tsne_fit(tsne, distances, precomputed_sigmas=None, skip_num_points=0, verbose=False):
    
    """Code adapted from the scikit-learn package: https://scikit-learn.org/"""
    
    assert tsne.metric == "precomputed"
    #if tsne.metric == "precomputed":
    if isinstance(tsne.init, str) and tsne.init == "pca":
        raise ValueError(
            'The parameter init="pca" cannot be used with metric="precomputed".'
        )
        
    if tsne.method not in ["barnes_hut", "exact"]:
        raise ValueError("'method' must be 'barnes_hut' or 'exact'")
    if tsne.angle < 0.0 or tsne.angle > 1.0:
        raise ValueError("'angle' must be between 0.0 - 1.0")
    
    if tsne.learning_rate == "warn":
        # See issue #18018
        warnings.warn(
            "The default learning rate in TSNE will change "
            "from 200.0 to 'auto' in 1.2.",
            FutureWarning,
        )
        tsne._learning_rate = 200.0
    else:
        tsne._learning_rate = tsne.learning_rate
    if tsne._learning_rate == "auto":
        # See issue #18018
        tsne._learning_rate = distances.shape[0] / tsne.early_exaggeration / 4
        tsne._learning_rate = np.maximum(tsne._learning_rate, 50)
    else:
        if not (tsne._learning_rate > 0):
            raise ValueError("'learning_rate' must be a positive number or 'auto'.")
    tsne.learning_rate_ = tsne._learning_rate #for compatibility with different versions of scikit-learn

    if tsne.method == "barnes_hut":
        distances = tsne._validate_data(
            distances,
            accept_sparse=["csr"],
            ensure_min_samples=2,
            dtype=[np.float32, np.float64],
        )
    else:
        distances = tsne._validate_data(
            distances, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
        )

    check_non_negative(
        distances,
        "TSNE.fit(). With metric='precomputed', distances "
        "should contain positive distances.",
    )

    if tsne.method == "exact" and issparse(distances):
        raise TypeError(
            'TSNE with method="exact" does not accept sparse '
            'precomputed distance matrix. Use method="barnes_hut" '
            "or provide the dense distance matrix."
        )

    if tsne.method == "barnes_hut" and tsne.n_components > 3:
        raise ValueError(
            "'n_components' should be inferior to 4 for the "
            "barnes_hut algorithm as it relies on "
            "quad-tree or oct-tree."
        )
    random_state = check_random_state(tsne.random_state)

    if tsne.early_exaggeration < 1.0:
        raise ValueError(
            "early_exaggeration must be at least 1, but is {}".format(
                tsne.early_exaggeration
            )
        )

    if tsne.n_iter < 250:
        raise ValueError("n_iter should be at least 250")


    n_samples = distances.shape[0]
    
    if n_samples == distances.shape[1]: #sq distance matrix ("exact" method)
        P = my_joint_probabilities(distances, tsne.perplexity, precomputed_sigmas, verbose)
    else: #knn approximation
        P = my_joint_probabilities_nn(distances, tsne.perplexity, precomputed_sigmas, verbose)
    if tsne.method == "barnes_hut":
        P = sp.sparse.csr_matrix(squareform(P))
        val_P = P.data.astype(np.float32, copy=False)
        neighbors = P.indices.astype(np.int64, copy=False)
        indptr = P.indptr.astype(np.int64, copy=False)
      
    if isinstance(tsne.init, np.ndarray):
        distances_embedded = tsne.init
    elif tsne.init == "random":
        # The embedding is initialized with iid samples from Gaussians with
        # standard deviation 1e-4.
        distances_embedded = 1e-4 * random_state.randn(n_samples, tsne.n_components).astype(
            np.float32
        )
    else:
        raise ValueError("'init' must be 'random', or a numpy array")

    # Degrees of freedom of the Student's t-distribution. The suggestion
    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(tsne.n_components - 1, 1)

    tsne.embedding_ = tsne._tsne(
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded=distances_embedded,
        neighbors=None,
        skip_num_points=skip_num_points,)
    
    return P

def my_tsne_fit_transform(tsne, sqdistances, precomputed_sigmas=None, recompute=True, skip_num_points=0, verbose=False):
    """ Allows one to fit a TSNE object (from sklearn.manifold) using custom sigmas (kernel scales)"""
    
    if recompute or not hasattr(tsne, 'embedding_'):
        my_tsne_fit(tsne, sqdistances, precomputed_sigmas, skip_num_points, verbose)
    return tsne.embedding_

def spectral_init_from_umap(graph, m, random_state=123456789):
    
    """ Computes a spectral embedding (akin to Laplacian eigenmaps) for initializing t-SNE 
    (analogously to what is done in UMAP, for a fair comparison b/w the two).
    The code below was adapted from the `spectral_layout` function in the umap-learn package. """

    n_components, labels = sp.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        print(f'Found more than 1 connected component ({n_components}).')
        print('UMAP does additional pre-processing in this case.')
        print("Please run UMAP's spectral_layout for a fair comparison.")
        
    diag_data = np.asarray(graph.sum(axis=0))

    # Normalized Laplacian
    I = sp.sparse.identity(graph.shape[0], dtype=np.float64)
    D = sp.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = m + 1
    eigenvectors, eigenvalues, _ = sp.sparse.linalg.svds(L, k=k, return_singular_vectors='u',
                                                        random_state=random_state)
    order = np.argsort(eigenvalues)[1:k]
    return eigenvectors[:, order]

def computeTSNEsigmas(sqdistances, desired_perplexity, verbose=False):

    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    sqdistances = sqdistances.astype(np.float32, copy=False)
    return get_tsne_sigmas(sqdistances, desired_perplexity, verbose)

def computeTSNEkernel(D2, sigmas, normalize=True, symmetrize=True, return_sparse=True):
    
    """ Computes the t-SNE kernel from a dense square matrix of squared distances and 
    precomputed sigmas """

    n_samples = D2.shape[0]

    if sp.sparse.issparse(D2):
        D2.sort_indices()
        n_samples = D2.shape[0]
        distances_data = D2.data.reshape(n_samples, -1)
        distances_data = distances_data
    else:
        distances_data = D2

    conditional_P = np.zeros_like(distances_data)
    for i in range(conditional_P.shape[0]):
        conditional_P[i] = np.exp(-distances_data[i]/(2*sigmas[i]**2))
        conditional_P[i,i] = 0
        if normalize:
            sum_Pi = max(conditional_P[i].sum(),1e-8)
            conditional_P[i] /= sum_Pi

    if sp.sparse.issparse(D2):
        conditional_P = csr_matrix( (conditional_P.ravel(), distances.indices, distances.indptr),
                        shape=(n_samples, n_samples))
    if symmetrize:
        conditional_P = (conditional_P + conditional_P.T)
        if not normalize:
            conditional_P *= .5

    if normalize:

        # Normalize the joint probability distribution
        sum_P = np.maximum(conditional_P.sum(), np.finfo(D2.dtype).eps)
        conditional_P /= sum_P
        assert np.all(np.abs(P.data) <= 1)

    if return_sparse:
        conditional_P = sp.sparse.csr_matrix(conditional_P)

    return conditional_P
    