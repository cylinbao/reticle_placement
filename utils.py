import scipy.sparse as sp
import numpy as np
import torch

def load_scipy_csr(path, name):
    csr = sp.load_npz(f'{path}/{name}_graph.npz')
    return csr

def load_feat(path, name):
    feats = np.load(f'{path}/{name}_feat.npy')
    feats = torch.from_numpy(feats).float()
    return feats
