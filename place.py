import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import scipy as sp
import dgl
from utils import *
from model import ReticlePlaceNet

def get_place(logits, mask):
    _logits = torch.mul(logits, mask)
    prob, loc = _logits.max(0)
    mask[loc] = 0
    return logits[loc], loc, mask

def get_coord(loc, N):
    x = (loc % N).type(torch.int)
    y = torch.floor(loc / N).type(torch.int)
    return x, y

def loc_mul_p(x, y, p):
    return x*p, y*p

def cost_func(graph, probs, placements, pl_w):
    srcs, dsts = graph.edges()
    costs = []
    losses = []
    for (s, d) in zip(srcs, dsts):
        p_s = probs[s]
        loc_s = placements[s]
        p_d = probs[d]
        loc_d = placements[d]

        x_s, y_s = get_coord(loc_s, pl_w)
        x_d, y_d = get_coord(loc_d, pl_w)

        c = torch.pow(x_s - x_d, 2) + torch.pow(y_s - y_d, 2)
        costs.append(c)

        x_sp, y_sp = loc_mul_p(x_s, y_s, p_s)
        x_dp, y_dp = loc_mul_p(x_d, y_d, p_d)
        l = torch.pow(x_sp - x_dp, 2) + torch.pow(y_sp - y_dp, 2)
        losses.append(l)

    costs = torch.stack(costs)
    cost = torch.sum(costs)

    losses = torch.stack(losses)
    loss = torch.sum(losses)
        
    return loss, cost


def train(nepochs, nnode, feat_len, n_channel, n_width, graph, feat):
    model = ReticlePlaceNet(feat_len, n_channel, n_width)

    pl_w = model.f_width

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for e in range(nepochs):
        with torch.autograd.set_detect_anomaly(True):
            model.train()
            mask = torch.ones(pl_w * pl_w) #, dtype=torch.bool)

            probs = []
            placements = []

            for n in range(nnode):
            # for n in range(1):
                graph.ndata['h'] = feat
                logits = model(graph, n)
                p, loc, mask = get_place(logits, mask)
                probs.append(p)
                placements.append(loc)

                # update node feature
                x, y = get_coord(loc, pl_w)
                f_node = [1, 1, x, y]
                # g.ndata['feat'][n] = torch.tensor(f_node)
                feat[n] = torch.tensor(f_node)

            # print(probs)
            # print(placements)

            optimizer.zero_grad()
            loss, cost = cost_func(graph, probs, placements, pl_w)
            # print(loss.item(), cost.item())
            print("Epoch {}: loss = {:.6f}, cost = {}".format(e, loss, cost))
            loss.backward()
            # optimizer.step()


if __name__ == '__main__':
    n_channel = 8
    n_width = 2

    path = "./programs/mac"
    prog = "mac4"

    csr = load_scipy_csr(path, prog)
    feat = load_feat(path, prog)

    nnode, feat_len = feat.shape

    # construct dgl graph 
    g = dgl.from_scipy(csr)
    g = dgl.add_reverse_edges(g)
    g = dgl.add_self_loop(g)

    # g.ndata['feat'] = feat

    # train(500, nnode, feat_len, n_channel, n_width, g, feat)
    train(1, nnode, feat_len, n_channel, n_width, g, feat)
