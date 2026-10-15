import numpy as np
import scipy.linalg
from GraphProjections import *
import cvxpy as cp


# def generate_hypercube(dim:int, num_agents:int, stochastic:bool = False, weight_rule = "lap")->np.ndarray:
#     """ Generates a hypercube in dimension dim, with the number of vertices equal to num_agents
#     By default returns the adjacency matrix, but if stochastic is set to true will weight the matrix
#     to be a valid combination matrix

#     Args:
#         dim (int): dimension
#         num_agents (int): number of agents/vertices
#         stochastic (bool, optional): set graph to be row/col stochastic. Defaults to False.
#         weight_rule (str, optional): weighting rule if to be stochastic. Defaults to "lap".
#                                      options:   "lap": Laplacian maximum-degree rule,
#                                                 "met": metropolis rule
#                                                 "col": column stochastic (based on averaging)
#                                                 "row": row stochastic

#     Returns:
#         np.ndarray: adjacency matrix / policy matrix
#     """
#     assert(num_agents <= 2**dim)
        
#     H = np.zeros([num_agents,num_agents])
#     formatter = '{0:0'+str(dim)+'b}'
#     for k in range(num_agents):
#         for l in range(num_agents):
#             vl = np.array([int(i) for i in formatter.format(l)])
#             vk = np.array([int(i) for i in formatter.format(k)])
#             if np.linalg.norm(vl-vk)<=1:
#                 H[k,l] = 1
#     if not stochastic:
#         return H
    
#     match str.lower(weight_rule):
#         case "met": 
#             return metropolisWeight(H)
#         case "col":
#             return H/np.repeat(np.sum(H,0)[None,:],num_agents,0)
#         case "row":
#             return H/np.repeat(np.sum(H,1)[:,None],num_agents,1)
#         case _:
#             return laplacianWeight(H)
#     return H

def Dec2BinVec(num:int, num_bits:int=None)->np.array:
    """Converts a (decimal) integer into a binary vector

    Args:
        num (int): number to convert
        num_bits (int, optional): number of bits in output representation (vector length)

    Returns:
        np.array: binary vector
    """
    
    if num_bits is None:
        num_bits = int(np.ceil(np.log2(num)))
    else:
        num_bits = np.max([int(np.ceil(np.log2(num+1))), num_bits])
    
    formatter = '{0:0'+str(num_bits)+'b}'
    bin_vec = np.array([int(i) for i in formatter.format(num)])
    
    return bin_vec

def Dec2BaseVec(num:int, base:int, num_digits:int=None)->np.ndarray:
    """Converts a decimal integer into a vector in the specified base

    Args:
        num (int): decimal integer
        base (int): basis
        num_digits (_type_): numbe of digits (vector length)

    Returns:
        (np.ndarray): _description_
    """
    vec=np.array([int(d) for d in np.base_repr(num,base=base)])
    
    if num_digits is None:
        return vec
    if vec.shape[0]<num_digits:
        vec = np.pad(vec, [num_digits-vec.shape[0],0], 'constant', constant_values=0)
    return vec

    
def HammingDistance(num1:int, num2:int)->int:
    """Calculates the hamming distance between two (decimal) integers based on the
    binary representation of the integers
    """
    
    num_bits = int(np.ceil(np.max([np.log2(num1+1), np.log2(num2+1)])))
    
    num1_bin = Dec2BinVec(num1, num_bits)
    num2_bin = Dec2BinVec(num2, num_bits)
    
    return np.sum(np.abs(num1_bin-num2_bin))

def Hypercube(num_agents:int, stochastic:bool = False, weight_rule = "lap")->np.ndarray:
    """ Generates a hypercube in dimension dim, with the number of vertices equal to num_agents
    By default returns the adjacency matrix, but if stochastic is set to true will weight the matrix
    to be a valid combination matrix

    Args:
        num_agents (int): number of agents/vertices
        stochastic (bool, optional): set graph to be row/col stochastic. Defaults to False.
        weight_rule (str, optional): weighting rule if to be stochastic. Defaults to "lap".
                                     options:   "lap": Laplacian maximum-degree rule,
                                                "met": metropolis rule
                                                "col": column stochastic (based on averaging)
                                                "row": row stochastic

    Returns:
        np.ndarray: adjacency matrix / policy matrix
    """
    dim = int(np.ceil(np.log2(num_agents)))
        
    H = np.zeros([num_agents,num_agents])
    formatter = '{0:0'+str(dim)+'b}'
    for k in range(num_agents):
        H[k,k] = 1
        for l in range(num_agents):
            if HammingDistance(l,k)==1:
                H[k,l] = 1
    if not stochastic:
        return H
    
    # match str.lower(weight_rule):
    #     case "met": 
    #         return metropolisWeight(H)
    #     case "col":
    #         return H/np.repeat(np.sum(H,0)[None,:],num_agents,0)
    #     case "row":
    #         return H/np.repeat(np.sum(H,1)[:,None],num_agents,1)
    #     case _:
    #         return laplacianWeight(H)
    return H

def laplacianWeight(A:np.ndarray):
    """Returns weight matrix from the Laplacian maximum-degree rule, based on adjacency matrix A
    Assumes graph is connected

    Args:
        A (np.ndarray): adjacency matrix (KxK binary matrix)
    """
    K = A.shape[0]
    ONES = 1/K*np.ones([K,K])
    weight_mat = A*(ONES-1/K*np.eye(K,K))
    diag_elems = 1-np.sum(weight_mat,0)
    weight_mat += np.diag(diag_elems)
    return weight_mat

def metropolisWeight(A:np.ndarray):
    """Returns weight matrix from the metropolis rule, based on adjacency matrix A
    Assumes graph is connected

    Args:
        A (np.ndarray): adjacency matrix (KxK binary matrix)
    """
    K = A.shape[0]
    out_degrees = np.sum(A,1)[:,None]
    out_degrees = np.repeat(out_degrees, K, 1)
    out_degrees = out_degrees
    comparison_mat = 1/np.maximum(out_degrees,out_degrees.T)
    # off-diagonal elements are the maximum out degree between two neighbours
    weight_mat = (A-np.eye(K,K))*comparison_mat
    diag_elems = 1-np.sum(weight_mat,1)
    weight_mat += np.diag(diag_elems)
    return weight_mat


def graphLaplacian(A:np.ndarray, sum_idx:int=0):
    """Returns the graph laplacian for an adjacency matrix A

    Args:
        A (np.ndarray): adjacency matrix (KxK binary)
        sum_idx (int): index over which to sum - does not matter for undirected graphs (optional)
    """
    K = A.shape[0]
    L = -(np.ones([K,K])-np.eye(K,K))*A
    degrees = -np.sum(L,sum_idx)
    L += np.diag(degrees)
    return L


# def isConnected(A:np.ndarray)->bool:
#     """Check graph connectivity from adjacency. Checks whether second smallest eigenvalue is positive

#     Args:
#         A (np.ndarray): adjacency matrix (KxK binary)
#     """
#     L = graphLaplacian(A)
#     eigvals = np.sort(scipy.linalg.eigvals(L))
#     eigvals = np.round(eigvals,14)
#     return eigvals[1]>1e-14

# def isPrimitive(A:np.ndarray)->bool:
#     """Check whether graph is primitive from adjacency.

#     Args:
#         A (np.ndarray): adjacency matrix (KxK binary)
#     """
#     self_loop = (np.diag(A)==1).any()
#     connectivity = isConnected(A)
#     return (self_loop and connectivity)
    

def ErdosRenyiGraph(num_vert:int, prob:float, connected_graph:bool=False)->np.ndarray:
    """Construct Erdos Renyi random graph

    Args:
        num_vert (int): number of vertices 
        prob (float): edge probability (0<prob<1)
        connected_graph (bool): only return a connected graph
    """
    is_connected = False
    while not is_connected:
        G = np.random.uniform(low=0,high=1, size=[num_vert,num_vert])
        G = np.triu(G,1) #remove lower triangular values
        G = G+G.T + np.eye(num_vert,num_vert)
        G = 1*(G>1-prob)
        is_connected = isConnected(G)
        if not connected_graph:
            break
    
    return G

# def graphDiameter(G:np.ndarray)->int:
#     """Floyd-Warshall algorithm for determining graph diameter
#     https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm

#     Args:
#         G (np.ndarray):adjacency matrix (KxK binary)

#     Returns:
#         int: graph diameter (inf for non-connected graph)
#     """

#     num_vertices = G.shape[0]
#     distances = np.inf*np.ones([num_vertices,num_vertices])

#     for k in range(num_vertices):
#         for j in range(num_vertices):
#             if G[k,j] != 0:
#                 distances[k,j] = G[k,j]
#         distances[k,k] = 0

#     for k in range(num_vertices):
#         for j in range(num_vertices):
#             for i in range(num_vertices):
#                 if distances[i,j]>distances[i,k]+distances[k,j]:
#                     distances[i,j] = distances[i,k]+distances[k,j]
#     diameter = np.max(distances)
#     return diameter


def SLEM_Optimal(H:np.ndarray, force_pos:bool=False):
    """ Generates the weight matrix with the fastest mixing rate on a graph
    See: S. Boyd, P. Diaconis, and L. Xiao, “Fastest Mixing Markov Chain on a Graph,” 
         SIAM Rev., vol. 46, no. 4, pp. 667–689, 2004, doi: 10.1137/s0036144503423264.
  
    Args:
        H (np.ndarray): adjacency matrix for graph (KxK binary)
        force_pos (bool, optional): enforce positive weights. Defaults to False.
        force_constraint (bool,optional): force symmetry (by projection) and double stochasticity (by diagonal adjustment). Defaults to True

    Returns:
        np.ndarray: weight matrix
        float: objective value from solver
    """
    
    K = H.shape[0]

    # Construct the problem.
    P = cp.Variable([K,K])
    objective = cp.Minimize(cp.norm2(P-1/K*np.ones([K,K])))

    # note in CVXPY >> is overloaded to mean positive semidefinite
    ONE = np.ones([K,1])
    constraints = [ P@ONE==ONE, P==P.T, P[H==0]==0]
    if force_pos:
        constraints.append(P>=0)
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    return P.value



# def hypercube_FT_sequence(dim:int, num_agents:int)->np.ndarray:
# def FTC_seq_hyp(num_agents:int)->np.ndarray:
#     """Generate a sequence of combination matrices on a hypercube whose product is a scaled all ones matrix

#     Args:
#         dim (int): dimension of hypercube
#         num_agents (int): number of agents (should be 2^dim for exact answer)

#     Returns:
#         np.ndarray: 3D tensor, with each matrix stacked on the third dimension
#     """
#     G = Hypercube(num_agents,False)
#     W = Hypercube(num_agents,True,"met")

#     eig = np.round(np.sort(np.linalg.eigvalsh(W)),14)
#     eig = np.unique(eig)

#     num_eigs = eig.shape[0]

#     mat_sequence = np.zeros([num_agents,num_agents,num_eigs-1])

#     for s in range(num_eigs-1):
#         mat = 1/(1-eig[s])*(W - eig[s]*np.eye(num_agents,num_agents))
#         mat_sequence[:,:,s] = stochasticityTweak(projSparsSymmStoch_cyc(mat, G))

#     return mat_sequence

def FTC_seq_hyp(K:int)->np.ndarray:
    """FTC sequence for a power of 2 hypercube, where each matrix is spectrally constrained to 1
    Derived independently but turns out to be a specific case of the matrices in Nguyen et al 2023

    Args:
        K (int): number of agents

    Returns:
        np.ndarray: matrix [K x K x log2(K)]
    """
    tau = int(np.log2(K))
    A_dyn = np.zeros([K,K,tau])

    for t in range(tau):
        for k in range(K):  
            A_dyn[k,k,t] = 1/2
            for l in range(K):
                if HammingDistance(k,l)==1 and np.abs(k-l)==2**t:
                    A_dyn[k,l,t] = 1/2 
    return A_dyn



def onePeerExponential(K:int, sym:bool=False)->np.ndarray:
    """ Generate sequence of one-peer exponential graphs with finite time consensus
    K=2^tau nodes, and tau matrices in the sequence
    
    Args:
        K (int): number of nodes
        sym (bool, optional): make matrix symmetric (output will have two peers). Defaults to False
    
    Returns:
        (np.ndarray): sequence of finite time consensus matrices [real K x K x tau]
    """ 
    
    tau = np.ceil(np.log2(K)).astype(int)
    W = np.zeros([K,K,tau])

    for t in range(tau):
        for i in range(K):
            for j in range(K):
                if (j-i)%K==2**(t):
                    W[i,j,t] = 1/2
        W[:,:,t] += 1/2*np.eye(K,K)
        if sym:
            W[:,:,t] = 1/2*W[:,:,t]+1/2*W[:,:,t].T
        
    return W


def FTC_seq_nulling(W:np.ndarray, precision:float )->np.ndarray:
    """Generates the sequence of matrices for finite time consensus based on the successive nulling of eigenvalues
    procedure in Safavi  and Khan https://doi.org/10.1109/LSP.2014.2346657. The method relies on an eigendecomposition
    of the weight matrix and then distinguishing the unique eigenvalues

    Args:
        W (np.ndarray): any real, symmetric doubly-stochastic weight matrix with a simple eigenvalue at 1 [real K x K]
        precision (float): the number of decimal points to consider when determining if two eigenvalues are equal

    Returns:
        np.ndarray: sequence of FTC matrices (real K x K x S)
    """
    assert (W==W.T).all(), "Non-symmetric weight matrix"
    assert (W==np.real(W)).all(), "Complex weight matrix"

    K = W.shape[0]
    eigs = np.real(np.linalg.eigvals(W))
    idces,_ = np.unique(np.round(eigs, precision), return_index=True)[::-1]
    eigs = np.sort(eigs[idces])[::-1] #sorted in descending order
    
    assert eigs[1] < 1, "Non-simple eigenvalue at 1"
        
    W_seq = np.zeros([K,K,len(eigs)])
    for idx,_ in enumerate(eigs):
        
        if idx==0:
            a = 1/np.prod(eigs[1::]-1) + 1
        else:
            a = eigs[idx]
        W_seq[:,:,idx] = a*np.eye(K,K)-W
        
    return W_seq
    

import mpmath
def FTC_seq_eigrule(W:np.ndarray, round_factor)->np.ndarray:
    """Generates the sequence of matrices for finite time consensus based on eigenvalue rule (Vlaski)

    Args:
        W (np.ndarray): any real, symmetric doubly-stochastic weight matrix with a simple eigenvalue at 1 [real K x K]

    Returns:
        np.ndarray: sequence of FTC matrices (real K x K x S)
    """
    # K = W.shape[0]
    # eigs = np.real(np.linalg.eigvals(W))
    # idces,_ = np.unique(np.round(eigs, precision), return_index=True)[::-1]
    # eigs = np.sort(eigs[idces])[::-1] #sorted in descending order
    # eigs=eigs[1::]
    # assert eigs[0] < 1, "Non-simple eigenvalue at 1"
        
    # W_seq = np.zeros([K,K,len(eigs)])
    # for idx,_ in enumerate(eigs):
    #     a = eigs[idx]
    #     W_seq[:,:,idx] = 1/(1-a)*(W - a*np.eye(K,K))
    K = W.shape[0]
    G = 1*(W!=0)
    W = mpmath.matrix(W)
    E = mpmath.eigsy(W,eigvals_only=True)
    E = np.array(E, dtype=np.float128)
    W = np.array(laplacianWeight(G), dtype=np.float128)

    E = np.sort(E)
    _,idx = np.unique(np.round(round_factor*E,1),return_index=True)
    E = E[idx]

    # p = np.eye(K,K)
    s_idx = 0
    e_idx = len(E)-1
    A_dyn_eig = np.zeros([K,K,e_idx-s_idx])
    for k in np.arange(s_idx,e_idx):
        A_dyn_eig[:,:,k] = 1/(1-E[k])*(W - E[k]*np.eye(K,K))
        
    return A_dyn_eig
    


def RingGraph(K:int)->np.ndarray:
    """Generate the adjacency matrix for a ring graph with K vertices (self-loops included)

    Args:
        K (int): number of vertices

    Returns:
        np.ndarray: 0-1 adjacency matric
    """
    G = np.eye(K,K)+np.eye(K,K,-1)+np.eye(K,K,1)
    G[0,K-1] = 1
    G[K-1,0] = 1
    
    return G

def PathGraph(K:int)->np.ndarray:
    """Generate the adjacency matrix for a path graph with K vertices (self-loops included)

    Args:
        K (int): number of vertices

    Returns:
        np.ndarray: 0-1 adjacency matric
    """
    G = np.eye(K,K)+np.eye(K,K,-1)+np.eye(K,K,1)
    
    return G

def StarGraph(K:int)->np.ndarray:
    """Generate the adjacency matrix for a star graph with K vertices (self-loops included)

    Args:
        K (int): number of vertices

    Returns:
        np.ndarray: 0-1 adjacency matric
    """
    G  = np.eye(K,K)
    G[1::,0] = 1
    G[0,1::] = 1
    
    return G

def LatticeGraph(K:int)->np.ndarray:
    """Generates a square lattice on a sqrt(K)xsqrt(K) lattice.
    Rule is similar to that of hypercube, based on observation

    Args:
        K (int): number of nodes, must be a square number

    Returns:
        (np.array): adjacency matrix
    """
    base = int(np.sqrt(K))
    d = int(np.ceil(np.emath.logn(base,K)))
    G = np.zeros([K,K])
    for k in range(K):
        G[k,k] = 1
        for j in range(K):
            k_b = Dec2BaseVec(k,base,d)
            j_b = Dec2BaseVec(j,base,d)
            if np.sum(np.abs(k_b-j_b))==1:
                G[k,j] = 1
    return G
 
    
def FTC_seq_ring(K:int)->np.ndarray:
    """Generate a finite time sequence for a ring graph based on the eigenvalue rule.
    Uses the closed-form expression for the eigenvalues of a ring graph 

    Args:
        K (int): number of vertices

    Returns:
        np.ndarray: FTC sequence [real K x K x tau]
    """
    G = RingGraph(K).astype(np.longdouble)
    W = np.longdouble(1/3)*G


    eigs=[]
    for k in np.arange(2,K+1):
        eigs.append( np.longdouble(1/3+2/3*np.cos(2*np.pi*(k-1)/K)))

    eigs = np.array(eigs, dtype=np.longdouble)

    eigs = np.unique(np.round(eigs,15))
    A = np.zeros([K,K,len(eigs)])
    for idx, eig in enumerate(eigs):
        A[:,:,idx] = 1/(1-eig)*(W-eig*np.eye(K,K))
        A[:,:,idx] = stochasticityTweak(A[:,:,idx])
    
    return A


def FTC_seq_path(K:int)->np.ndarray:
    """Generate a finite time sequence for a path graph based on the eigenvalue rule.
    Uses the closed-form expression for the eigenvalues of a path graph 

    Args:
        K (int): number of vertices

    Returns:
        np.ndarray: FTC sequence [real K x K x tau]
    """
    adjacency = PathGraph(K)-np.eye(K,K)
    degree_mat = np.diag(np.sum(adjacency,1))
    laplacian = degree_mat - adjacency

    beta = 1/4 #this parameter does not affect set of eigenvalues for FTC matrices
    weight_mat = np.eye(K,K)-beta*laplacian

    eigs = [1-2*beta+2*beta*np.cos(np.pi*(k-1)/K) for k in np.arange(1,K+1) ]
    eigs = np.sort(np.unique(np.array(eigs)))

    A = np.zeros([K,K,len(eigs)-1])

    for s in range(len(eigs)-1):
        A[:,:,s] = 1/(1-eigs[s])*(weight_mat-eigs[s]*np.eye(K,K))
        A[:,:,s] = stochasticityTweak(A[:,:,s])
        
    return A

def FTC_seq_star(K:int)->np.ndarray:
    """Generate a finite time sequence for a star graph based on the eigenvalue rule.
    Uses the closed-form expression for the eigenvalues of a path graph 

    Args:
        K (int): number of vertices

    Returns:
        np.ndarray: FTC sequence [real K x K x tau]
    """
    adjacency = StarGraph(K)-np.eye(K,K)
    degree_mat = np.diag(np.sum(adjacency,1))
    laplacian = degree_mat - adjacency

    beta = 1/(K+1) 
    weight_mat = np.eye(K,K)-beta*laplacian

    A = np.zeros([K,K,2])

    A[:,:,0] = 1/(beta)*(weight_mat-(1-beta)*np.eye(K,K))
    A[:,:,0] = stochasticityTweak(A[:,:,0])
    A[:,:,1] = 1/(beta*K)*(weight_mat-(1-beta*K)*np.eye(K,K))
    A[:,:,1] = stochasticityTweak(A[:,:,1])
    
    return A

def FTC_seq_path2(K:int)->np.ndarray:
    """Generate finite time consensus sequence for a path with 2^n agents.
    Based on a recursive formula.

    Args:
        K (int): number of agents, power of 2

    Returns:
        (np.ndarray): FTC sequence of size [K x K x K-1]
    """
    assert np.log2(K)-np.ceil(np.log2(K))==0, "Number of nodes not a power of 2"
    
    #Initialisation
    FTC_Seq = np.zeros([K,K,K-1])
    ONES2 = np.ones([2,2]) #2x2 all ones matrix
    EXCH2 = np.fliplr(np.eye(2,2)) #2x2 echange matrix (anti-diagonal matrix)
    
    if K==2:
        return 1/2*ONES2[:,:,None]
    if K == 4:
        FTC_Seq[:,:,0] = scipy.linalg.block_diag(ONES2/2, ONES2/2)
        FTC_Seq[:,:,1] = scipy.linalg.block_diag(np.eye(1), EXCH2, np.eye(1))
        FTC_Seq[:,:,2] = scipy.linalg.block_diag(ONES2/2, ONES2/2)
        return FTC_Seq
    
    Prev_Seq = FTC_seq_path2(int(K/2))
    #Block diagonals for FTC seq for path that is half the size
    for t in range(int(K/2)-1):
        FTC_Seq[:,:,t] = scipy.linalg.block_diag(Prev_Seq[:,:,t], Prev_Seq[:,:,t])
    #Permutation matrices
    for t in np.arange(int(K/2)-1, K-1):
        size_identity_mat = K-t-2
        num_exchange_mats = int((K-2*size_identity_mat)/2)
        list_mats = [np.eye(size_identity_mat)]
        [list_mats.append(EXCH2) for i in range(num_exchange_mats)]
        list_mats.append(np.eye(size_identity_mat))
        FTC_Seq[:,:,t] = scipy.linalg.block_diag(*list_mats)
    #Pairwise averaging matrix (copy first matrix in sequence which is always averaging)
    FTC_Seq[:,:,K-2] = FTC_Seq[:,:,0].copy()

    return FTC_Seq


def FTC_seq_ring2(K:int)->np.ndarray:
    """FTC sequence for a ring of size 2^D based on alternating averaging and exchange steps

    Args:
        K (int): number of nodes (must be a power of 2)

    Returns:
        np.ndarray: FTC seqeunce [K x K x K/2]
    """
    assert np.log2(K)-np.ceil(np.log2(K))==0, "Number of nodes not a power of 2"
    
    #Initialisation of averaging matrices and exchange matrices
    ONES2 = 1/2*np.ones([2,2])

    AVG1 = scipy.linalg.block_diag(*[ONES2 for i in range(int(K/2))]) 
    AVG2 = np.roll(np.roll(AVG1,-1,0),-1,1)
    AVG_MATS = [AVG1, AVG2]

    EXCH1 = 1*(AVG1-1/2*np.eye(K,K) > 0)
    EXCH2 = 1*(AVG2-1/2*np.eye(K,K) > 0)
    EXCH_MATS = [EXCH1, EXCH2]

    # FTC sequence generated using alternating averaging and exchange steps
    # 1 round averaging + (2^0-1) rounds of exchanges, 
    # 1 round averaging + (2^1 -1) rounds exchanges, etc. 
    # we alternate between which neighbours we are communicating with using the pointers
    num_exchanges = 1
    consensus_num = int(K/2)
    FTC_seq = []

    avg_pointer = 0
    exch_pointer = 0
    
    while np.shape(FTC_seq)[0]<consensus_num-1:
        # one averaging step
        FTC_seq.append(AVG_MATS[avg_pointer].copy()) 
        exch_pointer = (avg_pointer+1)%2 
        # multiple exchange steps
        for _ in range(num_exchanges-1):
            FTC_seq.append(EXCH_MATS[exch_pointer].copy())
            exch_pointer = (exch_pointer+1)%2
        avg_pointer = (exch_pointer)    
        num_exchanges = num_exchanges*2
    
    FTC_seq.append(AVG_MATS[avg_pointer])

    FTC_seq = np.concatenate([A[:,:,None] for A in FTC_seq],2)
    
    return FTC_seq



def FTC_seq_grid2(num_rows:int, num_cols:int)->np.ndarray:
    """ Generate the FTC sequence for a grid where each dimension is a power of 2.
    Note that the nodes in the grid are numbered in row major format
    The FTC sequence is built by considering each row as a path and stacking the FTC sequences
    for the rows, and then considering each column as a path, stacking the column FTC sequences and 
    transforming the column FTC into row major format. The consensus number, tau, is the consensus number
    for the horizontal path + the vertical path ie. tau = num_rows - 1 + num_cols - 1

    Args:
        num_rows (int): number of rows in the grid 
        num_cols (int): number of columns in the grid

    Returns:
        np.ndarray: FTC sequence [(num_rows*num_cols) x (num_rows*num_cols) x tau ]
    """

    tmp_FTC = FTC_seq_path2(num_cols)
    HorizontalFTC = np.zeros([num_rows*num_cols, num_rows*num_cols, tmp_FTC.shape[2]])
    for t in range(tmp_FTC.shape[2]):
        HorizontalFTC[:,:,t] = scipy.linalg.block_diag(*[tmp_FTC[:,:,t] for i in range(num_rows)])

    TransfMat = np.zeros([num_cols*num_rows, num_cols*num_rows])
    for r in range(num_rows):
        for c in range(num_cols):
            col_numbering = r+c*num_rows
            row_numbering = c + r*num_cols
            TransfMat[col_numbering,row_numbering] = 1

    tmp_FTC = FTC_seq_path2(num_rows)

    VerticalFTC = np.zeros([num_rows*num_cols, num_rows*num_cols, tmp_FTC.shape[2]])
    for t in range(tmp_FTC.shape[2]):
        stacked_mat = scipy.linalg.block_diag(*[tmp_FTC[:,:,t] for i in range(num_cols)])
        VerticalFTC[:,:,t] = TransfMat.T@stacked_mat@TransfMat

    FTC_seq = np.concatenate([HorizontalFTC, VerticalFTC], axis = 2)
    return FTC_seq



def GridGraph(num_rows:int, num_cols:int)->np.ndarray:
    """ Adjacency matrix for a grid. Note that the nodes in the grid are numbered in row major format.
    Works by generating a paths horizontally along the grid and paths vertically along the grid, converting the 
    numbering from column-major in the veryical to row-major and then adding the horizontal and vertical adjacencies

    Args:
        num_rows (int): number of rows in the grid 
        num_cols (int): number of columns in the grid

    Returns:
        np.ndarray: Adjacency matrix[(num_rows*num_cols) x (num_rows*num_cols) ]
    """
    horizontal_path = PathGraph(num_cols)
    G_horizontal = scipy.linalg.block_diag(*[horizontal_path for i in range(num_rows)])
    
    TransfMat = np.zeros([num_cols*num_rows, num_cols*num_rows])
    for r in range(num_rows):
        for c in range(num_cols):
            col_numbering = r+c*num_rows
            row_numbering = c + r*num_cols
            TransfMat[col_numbering,row_numbering] = 1
    vertical_path = PathGraph(num_rows)
    G_vertical = scipy.linalg.block_diag(*[vertical_path for i in range(num_cols)])
    G_vertical = TransfMat.T@G_vertical@TransfMat
    
    return G_horizontal+G_vertical-np.eye(num_cols*num_rows, num_cols*num_rows)



def HelmGraph(L:int)->np.ndarray:
    """Helm graph of wheel size L

    Args:
        L (int): wheel size

    Returns:
        np.ndarray: adjacency matrix [2L+1 x 2L+1]
    """

    K = 2*L+1

    G = np.zeros([K,K])

    for k in np.arange(1,L+1):
        G[0,k] = 1
        G[k,0] = 1
        G[k,k%L+1] = 1
        G[k%L+1,k] = 1
        G[k,k+L] = 1
        G[k+L,k] = 1
    G = G+np.eye(K,K)
    G = 1*(G>0)
    return G


def BarbellGraph(K:int)->np.ndarray:
    """Generate barbell graph with K agents

    Args:
        K (int): number of agents, must be even

    Returns:
        np.ndarray: adjacency matrix [K x K]
    """
    assert K%2==0, 'No. agents must be even'
    L = int(K/2)
    G = scipy.linalg.block_diag(np.ones([L,L]), np.ones([L,L]))
    G[0,L] = 1
    G[L,0] = 1
    return G



def FTC_seq_barbell(K:int)->np.ndarray:
    """ Generate a nonnegative FTC sequence for the barbell graph. May/may not be optimal.

    Args:
        int (K): number of agents, must be multiple of 4 and greater than or equal to 8

    Returns:
        np.ndarray: FTC sequence [K x K x K/2+1]
    """
    assert K%4 == 0, 'No. agents must be multiple of 4'
    assert K>=8, 'No. agents must be greater than or equal to 8'
    
    L = int(K/2)
    A = np.zeros([K,K,L+1])

    ONES_L = 1/L*np.ones([L,L])
    EXCH_MID = np.eye(K,K)
    EXCH_MID[0,0] = 0
    EXCH_MID[0,L] = 1
    EXCH_MID[L,0] = 1
    EXCH_MID[L,L] = 0

    A[:,:,0] = scipy.linalg.block_diag(ONES_L, ONES_L)
    A[:,:,1] = EXCH_MID.copy()

    for i in range(int(L/2-1)):
        exch = np.eye(L,L)
        exch[0,0] = 0
        exch[0,i+1] = 1
        exch[i+1,0] = 1
        exch[i+1,i+1] = 0
        A[:,:,2+2*i] = scipy.linalg.block_diag(exch, exch)
        A[:,:,2+2*i+1] = EXCH_MID.copy()
    A[:,:,-1] = scipy.linalg.block_diag(ONES_L, ONES_L)
    
    return A



def XiaoAndBoydGraph()->np.array:
    G = np.zeros([8,8])
    G[1:5,0] = 1
    G[0,1:5] = 1
    
    for k in np.arange(1,7):
        G[k,k+1] = 1
        G[k+1,k] = 1
    G[1,4::] = 1
    G[4::,1] = 1
    G[4,6] = 1
    G[6,4] = 1
    G[4:6,7] = 1
    G[7,4:6] = 1
    G = G+np.eye(8,8)
    return G

def SunletGraph(L:int)->np.ndarray:
    """Sunlet graph of wheel size L

    Args:
        L (int): wheel size

    Returns:
        np.ndarray: adjacency matrix [2L x 2L]
    """

    K = 2*L

    G = np.zeros([K,K])

    for k in range(L):
        G[k,(k+1)%L] = 1
        G[(k+1)%L,k] = 1
        G[k,k+L] = 1
        G[k+L,k] = 1
    G = G+np.eye(K,K)
    G = 1*(G>0)
    return G

