import argparse
import scipy.sparse as sp
import numpy as np

def gen_mac_edges(n_in):
    edges = []

    s_base = 0
    d_base = n_in
    nnode = n_in
    tot = (n_in*2) - 1

    while(nnode > 1):
        next_nnode = int(nnode/2)
        for i in range(next_nnode):
            d = (d_base + i)
            s = ((d_base + i) % nnode)*2 + s_base
            e1, e2 = (s, d), (s+1, d)
            edges.append(e1)
            edges.append(e2)
        s_base += nnode
        d_base += next_nnode
        nnode = next_nnode

    return edges, tot


def edges_to_csr(edges, tot):
    src, dst = map(list, zip(*edges))

    src = np.array(src)
    dst = np.array(dst)
    data = np.ones(src.shape[0], dtype=int)

    csr = sp.csr_matrix((data, (src, dst)), shape=(tot, tot))
    return csr


def gen_feat(n_tot):
    types = np.ones(n_tot, dtype=int)
    slices = np.ones(n_tot, dtype=int)
    x = np.random.randint(32, size=n_tot)
    y = np.random.randint(32, size=n_tot)

    feat = np.array([types, slices, x, y]).T
    return feat
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-in', type=int, default=8, metavar='N',
                        help='number of input nodes, must be power of 2')

    args = parser.parse_args()

    fname = f"./programs/mac/mac{args.num_in}"
    edges, tot = gen_mac_edges(args.num_in)
    csr = edges_to_csr(edges, tot)
    sp.save_npz(f"{fname}_graph.npz", csr)

    feat = gen_feat(tot)
    np.save(f"{fname}_feat.npy", feat)

    print("MAC program generation finished.")
