import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class EdgeGraphConv(nn.Module):
    def __init__(self,
                 num_hidden,
                 activation=None):
        super(EdgeGraphConv, self).__init__()

        self.edge_fc = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.node_fc = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = activation

    def edge_embedding(self, edges):
        eh = torch.cat([edges.src['h'], edges.dst['h']], 1)
        eh = self.edge_fc(eh)

        if self.act != None:
            eh = self.act(eh)
        return {'eh':  eh}

    def forward(self, g, feat):
        g.ndata['h'] = feat
        g.update_all(self.edge_embedding, fn.mean('eh', 'h'))

        rst = self.node_fc(g.ndata['h']) 

        # degs = g.in_degrees().float().clamp(min=1)
        # norm = 1.0 / degs
        # shp = norm.shape + (1,) * (feat.dim() - 1)
        # norm = torch.reshape(norm, shp)

        # # normalize the output features
        # rst = rst * norm

        if self.act != None:
            rst = self.act(rst)

        return rst
