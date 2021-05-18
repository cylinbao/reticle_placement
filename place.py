import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy as sp
import dgl

class PolicyNet(nn.Module):
    def __init__(self,
                 n_channels,
                 n_width):
        super(PolicyNet, self).__init__()
        self.linear0 = nn.Linear(n_channels, 
                        n_width*n_width*n_channels)
        self.deconvs = nn.ModuleList()
        self.n_channels = n_channels
        self.n_width = n_width

        _nc = n_channels 
        self.f_width = n_width
        while _nc > 1:
            self.deconvs.append(
                nn.ConvTranspose2d(_nc, int(_nc/2), 2, stride=2, 
                padding=0))
            _nc = int(_nc/2)
            self.f_width *= 2

    def forward(self, h):
        h = self.linear0(h)
        h = F.relu(h)
        h = h.view(1, self.n_channels, self.n_width, self.n_width)
        for i, deconv in enumerate(self.deconvs):
            h = deconv(h)
            if i < (len(self.deconvs) - 1):
                h = F.relu(h)
                
        h = F.softmax(h.flatten())
        h = h.view(self.f_width, self.f_width)
        return h


def get_place(logits, mask):
    logits = logits * mask
    place_loc = np.unravel_index(logits.argmax(), logits.shape)
    mask[place_loc] = 0
    return place_loc, mask


def cal_loss(graph, pl_assigns):
    srcs, dsts = graph.edges()
    cost = 0
    for (s, d) in zip(srcs.numpy(), dsts.numpy()):
        x0, y0 = pl_assigns[s] 
        x1, y1 = pl_assigns[d] 
        cost += (abs(x0 - x1) + abs(y0 - y1))

    return torch.empty(4, 4).fill_(cost)


def train(nepochs, nnode, feat_len, n_width, graph, feats):
    model = PolicyNet(feat_len, n_width)
    pl_w = model.f_width

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for e in range(nepochs):
        model.train()
        mask = torch.ones(pl_w, pl_w, dtype=torch.bool)
        pl_assigns = {}

        for n in range(nnode):
            logits = model(feats[n])
            ploc, mask = get_place(logits, mask)
            pl_assigns[n] = ploc
        print(pl_assigns)

        optimizer.zero_grad()
        loss = cal_loss(graph, pl_assigns)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    nnode = 8
    feat_len = 2
    n_width = 2

    # construct a graph of an adder tree with four input adders
    src_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    dst_ids = torch.tensor([4, 4, 5, 5, 6, 6, 7])

    g = dgl.graph((src_ids, dst_ids), num_nodes=nnode)
    # g = dgl.add_reverse_edges(g)
    # g = dgl.add_self_loop(g)

    feats = torch.randn(nnode, feat_len)

    train(1, nnode, feat_len, n_width, g, feats)
