import numpy as np
import networkx as nx
def google_matrix(G, alpha=0.85, personalization=None,nodelist=None, weight='weight', dangling=None):
    
    if nodelist is None:
        nodelist = G.nodes()

#Return the graph adjacency matrix as a NumPy matrix.
    M = nx.to_numpy_matrix(G, nodelist=nodelist, weight=weight)
    N = len(G)
    
    if N == 0:
        return M

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N) #np.repeat(3, 4) ==> array([3, 3, 3, 3])
    else:
         missing = set(nodelist) - set(personalization)
         if missing:
             raise NetworkXError('Personalization vector dictionary ''must have a value for every node. ''Missing nodes %s' % missing)
         
         p = np.array([personalization[n] for n in nodelist], dtype=float)
         p /= p.sum()
         
         # Dangling nodes
         
    if dangling is None:
         dangling_weights = p
    else:
         missing = set(nodelist) - set(dangling)
         if missing:
             raise NetworkXError('Dangling node dictionary ''must have a value for every node. ''Missing nodes %s' % missing)
                 
    # Convert the dangling dictionary into an array in nodelist order
         dangling_weights = np.array([dangling[n] for n in nodelist],dtype=float)
         dangling_weights /= dangling_weights.sum()
             
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]
    
    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    for node in dangling_nodes:
        M[node] = dangling_weights
        
    M = M.sum(axis = 1) # Normalize rows to sum to 1 and e M is a stochastic matrix
    
    return alpha * M + (1 - alpha) * p

#print Google Matrix

#G=nx.barabasi_albert_graph(60,41)
#gm=google_matrix(G,0.4)
#print(gm)

def pagerank_numpy(G,alpha=0.85,personalization=None,weight='weight',dangling=None):
    
    if len(G) == 0:
        return {}
    M = google_matrix(G, alpha, personalization=personalization,weight=weight, dangling=dangling)
    
    # use numpy LAPACK solver
    #LAPACK can solve systems of linear equations, linear least squares problems, eigenvalue problems and singular value problems.
    #Eigenvalues are a special set of scalars associated with a linear system of equations (i.e., a matrix equation) that are sometimes also known as characteristic roots, characteristic values (Hoffman and Kunze 1971), proper values, or latent roots (Marcus and Minc 1988, p. 144).
    #Any non-zero vector with v1 = v2 solves this equation. Therefore, is an eigenvector of A corresponding to λ = 3, as is any scalar multiple of this vector. Thus, the vectors vλ=1 and vλ=3 are eigenvectors of A associated with the eigenvalues λ = 1 and λ = 3, respectively.
    
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = eigenvalues.argsort()
    # eigenvector of largest eigenvalue at ind[-1], normalized
    largest = np.array(eigenvectors[:, ind[-1]]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))

#print pagerank
G=nx.barabasi_albert_graph(60,41)
pr=pagerank_numpy(G,0.4)
print(pr) 
    
