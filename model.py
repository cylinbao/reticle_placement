import torch 
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self,
                 feat_len,
                 n_channels,
                 n_width):
        super(PolicyNet, self).__init__()
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
