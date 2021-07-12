import numpy as np
import dgl
from utils import *
import json

def np_get_coord(loc, grid_shape):
    N = grid_shape[0]
    x = (loc % N)
    y = np.floor(loc / N).astype(int)
    return x, y

def random_place(grid_shape, nnode, seed):
    locs = np.arange(grid_shape[0]*grid_shape[1])
    # np.random.seed(int(time.time()))
    np.random.seed(seed)
    np.random.shuffle(locs)
    return locs[:nnode].tolist()

def cal_cost(g, grid_shape, places):
    srcs, dsts = g.edges()
    cost = 0

    for (s, d) in zip(srcs, dsts):
        loc_s = places[s]
        loc_d = places[d]

        x_s, y_s = np_get_coord(loc_s, grid_shape)
        x_d, y_d = np_get_coord(loc_d, grid_shape)

        cost += np.power(x_s - x_d, 2) + np.power(y_s - y_d, 2)
    return cost.item()

def dgl_to_py_edges(dgl_edges):
    srcs = dgl_edges[0].tolist()
    dsts = dgl_edges[1].tolist()
    return (srcs, dsts)

if __name__ == '__main__':
    path = "./programs/mac"
    prog = "mac4"

    csr = load_scipy_csr(path, prog)

    g = dgl.from_scipy(csr)
    # g = dgl.add_reverse_edges(g)
    # g = dgl.add_self_loop(g)

    grid_shape = (32, 32)
    nnode = csr.shape[0] 

    ite = 50

    placements = {}
    for i in range(ite):
        locs = random_place(grid_shape, nnode, i)
        c = cal_cost(g, grid_shape, locs)
        plc = {'cost': c, 'locs': locs} 
        placements[i] = plc

    edges = dgl_to_py_edges(g.edges())
    
    data = {}
    data['nnode'] = nnode
    data['grid'] = grid_shape
    data['edges'] = edges
    data['placements'] = placements

    path = "./dataset/"
    fname = prog + "_dataset.json"
    with open(path+fname, 'w') as f:
        json.dump(data, f)
