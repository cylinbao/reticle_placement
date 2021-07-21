import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from layer import EdgeGraphConv

class EdgeGNN(nn.Module):
    def __init__(self,
                 num_layer,
                 num_feat,
                 num_hidden,
                 activation):
        super(EdgeGNN, self).__init__()

        self.feat_fc = nn.Linear(num_feat, num_hidden, bias=True)
        self.layers = nn.ModuleList()
        self.classify_fc = nn.Linear(num_hidden, 1, bias=True)
        
        for i in range(num_layer):
            self.layers.append(EdgeGraphConv(num_hidden, activation))

    def forward(self, g, feat):
        h = self.feat_fc(feat)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        h = torch.mean(h, 0)
        p = self.classify_fc(h)
        return p

class ReticlePlaceNet(nn.Module):
    def __init__(self,
                 feat_len,
                 n_channels,
                 n_width):
        super(ReticlePlaceNet, self).__init__()

        self.egc = EdgeGraphConv(feat_len)

        self.linear0 = nn.Linear(feat_len*2, 
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

    def forward(self, g, nid):
        g = self.egc(g)

        feat = g.ndata['h']
        f_mean = torch.mean(feat, 0)
        f_node = feat[nid]
        h = torch.cat((f_mean, f_node))

        h = self.linear0(h)
        h = F.relu(h)
        h = h.view(1, self.n_channels, self.n_width, self.n_width)
        for i, deconv in enumerate(self.deconvs):
            h = deconv(h)
            if i < (len(self.deconvs) - 1):
                h = F.relu(h)
                
        h = F.softmax(h.flatten(), dim=0)
        # h = h.view(self.f_width, self.f_width)
        return h

class PlaceNet(nn.Module):
    def __init__(self,
                 feat_len,
                 n_channels,
                 n_width):
        super(PlaceNet, self).__init__()
        self.linear0 = nn.Linear(feat_len*2, 
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

    def forward(self, feat, nid):
        # g = self.egc(g)
        # feat = g.ndata['h']
        f_mean = torch.mean(feat, 0)
        f_node = feat[nid]
        h = torch.cat((f_mean, f_node))

        h = self.linear0(h)
        h = F.relu(h)
        h = h.view(1, self.n_channels, self.n_width, self.n_width)
        for i, deconv in enumerate(self.deconvs):
            h = deconv(h)
            if i < (len(self.deconvs) - 1):
                h = F.relu(h)
                
        h = F.softmax(h.flatten(), dim=0)
        # h = h.view(self.f_width, self.f_width)
        return h
