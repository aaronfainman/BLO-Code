import numpy as np



def projColStoch_sp(X, G):
    """Column stochastic projection for weights within neighbourhood of each agents (implicit sparsity)
    X: matrix to project, G is graph adjacency"""
    K = X.shape[0]
    for k in range(K):
        indices = G[:,k]==1
        d_in = np.sum(G[:,k]).astype(int)
        d_in = d_in if d_in !=0 else 1
        X[indices,k] = X[indices,[k]]-1/d_in*np.sum(X[indices,[k]])+1/d_in
    return X


def projColStoch(X):
    "Euclidean projection for column stochasticity "
    K = X.shape[0]
    return X@(np.eye(K,K)-1/K*np.ones([K,K]))+1/K*np.ones([K,K])

def projRowStoch(X):
    "Euclidean projection for column stochasticity "
    K = X.shape[0]
    return (np.eye(K,K)-1/K*np.ones([K,K]))@X+1/K*np.ones([K,K])


import cvxpy as cp
def projSymStochSpars_full(A, G):
    K = A.shape[0]
    ONE = np.ones([K,1])
    H = 1-G
    X = cp.Variable([K,K])
    objective = cp.Minimize(cp.sum_squares(A - X))
    constraints = []
    for i in range(K):
        constraints.append(np.diag(H[:,i])@X[:,i]==0)
    constraints.append(X-X.T==0)
    constraints.append(X@ONE==ONE)
    # constraints = [X@ONE==ONE, X-X.T==0]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    return X.value



def projSparsStochSymmTweak(A:np.ndarray, G:np.ndarray):
    """projection is sparsity -> stochastic -> symmetry -> diag tweak

    Args:
        A (np.ndarray): matrix to project
        G (np.ndarray): adjacency graph
    """
    K = A.shape[0]
    ONES = np.ones([K,K])
    X = G*A
    X = projColStoch_sp(X,G)
    X = 1/2*(X+X.T)
    X = stochasticityTweak(X)
    return X

def projSparsStochSymm(A:np.ndarray, G:np.ndarray):
    """projection is sparsity -> stochastic -> symmetry

    Args:
        A (np.ndarray): matrix to project
        G (np.ndarray): adjacency graph
    """
    K = A.shape[0]
    ONES = np.ones([K,K])
    X = G*A
    X = projColStoch_sp(X,G)
    X = 1/2*(X+X.T)
    return X

    
def projSparsSymmTweak(A:np.ndarray, G:np.ndarray):
    """projection is sparsity ->  symmetry -> tweak

    Args:
        A (np.ndarray): matrix to project
        G (np.ndarray): adjacency graph
    """
    K = A.shape[0]
    ONES = np.ones([K,K])
    X = G*A
    X = 1/2*(X+X.T)
    X = stochasticityTweak(X)
    return X


def projSymmStoch_true(A:np.ndarray)->np.ndarray:
    """True projection for a symmetric, doubly stochastic matrix (does not preserve sparsity)
    """
    K = A.shape[0]
    ONE = np.ones([K,1])

    X1 = 1/2*(A+A.T)
    X2 = projDoubleStoch(X1)
    # X2 = X1-1/K*X1@ONE@ONE.T-1/K*ONE@ONE.T@X1+1/K**2*ONE@ONE.T*np.sum(X1)+1/K*ONE@ONE.T
    
    return X2


def projDoubleStoch(A:np.ndarray)->np.ndarray:
    K = A.shape[0]
    ONE = np.ones([K,1])
    X = A-1/K*A@ONE@ONE.T-1/K*ONE@ONE.T@A+1/K**2*ONE@ONE.T*np.sum(A)+1/K*ONE@ONE.T
    return X


def projSparsSymmStoch_cyc(A:np.ndarray, G:np.ndarray, num_cycles=1000):
    """Cyclic projection for a sparse, symmetric, doubly stochastic matrix
    Returned matrix is always sparse, may have error in stochasticity

    Args:
        A (np.ndarray): matrix to project (real KxK)
        G (np.ndarray): sparsity pattern (binary KxK)
    """
    
    for cycle in range(num_cycles):
        A = G*projSymmStoch_true(A)
    return A
    


def stochasticityTweak(A:np.ndarray):
    """Tweaks diagonal elements to ensure that matrix is 
    Preserves symmetry

    Args:
        A (np.ndarray): matrix
    """
    K = A.shape[0]
    I = np.eye(K,K)
    X = (np.ones_like(A)-I)*A #set diagonal elements to 0
    off_diag_sums = np.sum( X, 0)
    diag_elems = (1-off_diag_sums)*I
    X += diag_elems
    return X


def projNSphere(x:np.ndarray, centre:np.ndarray, radius:float)->np.ndarray:
    """Projects point x onto the closest point on the surface of an N-sphere

    Args:
        x (np.ndarray): point [Mx1]
        centre (np.ndarray): centre of sphere [Mx1]
        radius (float): sphere radius
    Returns:
        np.ndarray: projected point [Mx1]
    """
    assert(not np.all(x==centre))
    
    proj = radius*(x-centre)/np.linalg.norm(x-centre) + centre
    
    return proj
    
