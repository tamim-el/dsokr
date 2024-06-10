import numpy as np
from scipy.stats import ortho_group
from grakel import Graph
from sklearn.model_selection import train_test_split



def load_toy_DSOKR(n=50000, dx=2000,
                   dy=1000, dytrue=50,
                   r=0.5, sigma=1.0):
    """
        Load synthetic dataset
    """
    E = sigma * np.eye(dy)
    eps = np.random.multivariate_normal(mean=np.zeros(dy), cov=E, size=n)

    spC = np.arange(dx) + 1
    spC = 1.0 / (spC.copy() ** r)
    UC = ortho_group.rvs(dx)
    C = UC.dot(np.diag(spC)).dot(UC.T)
    X = np.random.multivariate_normal(mean=np.zeros(dx), cov=C, size=n)

    U = ortho_group.rvs(dy)[:, :dytrue]
    H0 = np.random.normal(size=(dytrue, dx))
    H = U.dot(H0)

    Y_true = X.dot(H.T)
    Y = Y_true.copy() + eps

    return X, Y, Y_true


def to_grakels(Y_dict, n=None, do_edge_labels=False):
    """
    Convert a list of dictionaries containing adjacency matrices
    and one-hot-encoded label vectors to a Grakel Graph.
    
    Parameters
    ----------
    Y_dict : list.
        List of dictionaries including an adjacency matrix
        and a one-hot-encoded label vector.
    n : int, optional
        Number of graphs loaded. If None, all data are loaded.
        The default is None.
    edge_labels : bool, optional
        If True, edge labels are encoded in the Grakel instance.
        The default is False.

    Returns
    -------
    Y : 1-D array-like of size (n).
        Array of Grakel Graphs.

    """
    
    if n is None:
        n = len(Y_dict)
    
    Gs = list()

    for i in range(n):
        adj = Y_dict[i]['A']
        node_labels = np.argmax(Y_dict[i]['F'], axis=1)
        node_labels_dict = {j: label for (j, label) in enumerate(node_labels)}
        if do_edge_labels:
            edge_labels = np.argmax(Y_dict[i]['E'], axis=2)
            edges1, edges2 = np.where(adj == 1)
            edge_labels_dict = {(edges1[j], edges2[j]): edge_labels[edges1[j], edges2[j]] for j in range(len(edges1))}
        Gs.append(Graph(adj, node_labels=node_labels_dict, edge_labels=edge_labels_dict))
        
    Y = np.array(Gs)
        
    return Y


def load_smi2mol(path='Data/qm9/', n_valid=500, n_test=2000, delete_atoms=False, random_seed=42):
    """
    Load fingerprints to molecules data.

    Parameters
    ----------
    path : str, optional
        Path of the folder containing the pickle file. The default is 'Data/fingerprint2mol/'.
    n_tr : int, optional
        Number of training data loaded.
        The default is 60000.
    n_val : int, optional
        Number of validation data loaded.
        The default is 500.
    delete_atoms : bool, optional
        If True, remove entries corresponding to a graph of a single node.
        The default is False.
    smiles : bool, optional
        If True, smiles are returned as inputs rather than fingerprints.
        The default is False.

    Returns
    -------
    X_tr : 1-D array-like of size (n_tr) if smiles.
           2-D array-like of size (n_tr, 2048) otherwise.
        Training inputs as smiles if smiles == True,
        as fingerprints otherwise.
    Y_grkl_tr : 1-D array-like of size (n_tr).
        Training outputs as Graph instances.
    Y_dict_tr : list.
        Training outputs as dictionaries.
    X_val : 1-D array-like of size (n_val) if smiles.
           2-D array-like of size (n_val, 2048) otherwise.
        Training inputs as smiles if smiles == True,
        as fingerprints otherwise.
    Y_grkl_val : 1-D array-like of size (n_val).
        Validation outputs as Graph instances.
    Y_dict_val : list.
        Validation outputs as dictionaries.
    X_te : 1-D array-like of size (2000) if smiles.
           2-D array-like of size (2000, 2048) otherwise.
        Training inputs as smiles if smiles == True,
        as fingerprints otherwise.
    Y_grkl_te : 1-D array-like of size (2000).
        Test outputs as Graph instances.
    Y_dict_te : list.
        Test outputs as dictionaries.

    """
    X_tr_tmp = np.load(path + 'X_smiles_train_qm9.npy', allow_pickle=True)
    Y_dict_tr_tmp = np.load(path + 'y_train_qm9.npy', allow_pickle=True)
    X_te_tmp= np.load(path + 'X_smiles_test_qm9.npy', allow_pickle=True)
    Y_dict_te_tmp = np.load(path + 'y_test_qm9.npy', allow_pickle=True)

    X = np.concatenate((X_tr_tmp, X_te_tmp))
    Y_dict = np.concatenate((Y_dict_tr_tmp, Y_dict_te_tmp))
    Y_grkl = to_grakels(Y_dict, do_edge_labels=True)

    if delete_atoms:
        idx_atoms = []
        for i in range(len(X)):
            if Y_grkl[i].get_edges() == []:
                idx_atoms.append(i)

        X = np.delete(X, idx_atoms, axis=0)
        Y_dict = np.delete(Y_dict, idx_atoms)
        Y_grkl = np.delete(Y_grkl, idx_atoms)

    X_reste, X_te, Y_grkl_reste, Y_grkl_te, Y_dict_reste, Y_dict_te = train_test_split(
        X, Y_grkl, Y_dict, test_size=n_test, random_state=random_seed
    )

    X_tr, X_val, Y_grkl_tr, Y_grkl_val, Y_dict_tr, Y_dict_val = train_test_split(
        X_reste, Y_grkl_reste, Y_dict_reste, test_size=n_valid, random_state=random_seed
    )

    assert len(X_tr) == len(Y_grkl_tr)
    assert len(X_tr) == len(Y_dict_tr)
    assert len(X_val) == len(Y_grkl_val)
    assert len(X_val) == len(Y_dict_val)
    assert len(X_te) == len(Y_grkl_te)
    assert len(X_te) == len(Y_dict_te)

    return X_tr, Y_grkl_tr, Y_dict_tr, X_val, Y_grkl_val, Y_dict_val, X_te, Y_grkl_te, Y_dict_te


def load_text2mols(path='Data/chebi-20/'):
    """
    Load text to molecules data.

    Parameters
    ----------
    path : str, optional
        Path of the folder containing the pickle file. The default is 'Data/chebi-20/'.

    Returns
    -------
    X_tr : 1-D array-like of size (n_tr).
    Y_grkl_tr : 1-D array-like of size (n_tr).
        Training outputs as Graph instances.
    Y_dict_tr : list.
        Training outputs as dictionaries.
    X_val : 1-D array-like of size (n_val).
    Y_grkl_val : 1-D array-like of size (n_val).
        Validation outputs as Graph instances.
    Y_dict_val : list.
        Validation outputs as dictionaries.
    X_te : 1-D array-like of size (n_te).
    Y_grkl_te : 1-D array-like of size (n_te).
        Test outputs as Graph instances.
    Y_dict_te : list.
        Test outputs as dictionaries.
    Y_grkl_cand : 1-D array-like of size (n_cand).
       Graph candidate cinstances.
    Y_dict_cand : list.
       Graph candidate as dictionaries.
    """
    X_tr = np.load(path + 'X_train.npy', allow_pickle=True)
    Y_dict_tr  = np.load(path + 'y_train.npy', allow_pickle=True)
    Y_gkl_tr = to_grakels(Y_dict_tr, do_edge_labels=True)
    
    X_val = np.load(path + 'X_valid.npy', allow_pickle=True)
    Y_dict_val =  np.load(path + 'y_valid.npy', allow_pickle=True)
    Y_gkl_val = to_grakels(Y_dict_val, do_edge_labels=True)
    
    X_te = np.load(path + 'X_test.npy', allow_pickle=True)
    Y_dict_te = np.load(path + 'y_test.npy', allow_pickle=True)
    Y_gkl_te = to_grakels(Y_dict_te, do_edge_labels=True)

    Y_dict_cand = np.load(path + 'y_cand.npy', allow_pickle=True)
    Y_gkl_cand = to_grakels(Y_dict_cand, do_edge_labels=True)

    
    return X_tr, Y_gkl_tr, Y_dict_tr, X_val, Y_gkl_val, Y_dict_val, X_te, Y_gkl_te, Y_dict_te, Y_gkl_cand, Y_dict_cand